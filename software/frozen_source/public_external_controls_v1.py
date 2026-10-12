"""Post-result, zero-fit diagnostics and missing source-external controls.

All real pixels/weights are read on the server only. No target fitting, threshold
search, reference-driven inference crop or original54 re-evaluation is permitted.
The parent external evaluation remains immutable.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
import time
import traceback
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import torch

from . import public_external_v1 as parent
from .cascade import CascadeConfig, image_tensor, predict_regions, reconstruct
from .data import load_cached
from .evaluation import crop
from .round3 import open_rgb, REFINER_FAMILIES
from .round4 import global_spec, load_network, run_path, actual_environment
from .stats import bootstrap_mean_ci, paired_bootstrap, sign_flip_pvalue, holm_adjust, wilson_rate_ci
from .training import seed_all

NEW_METHODS = ("global_deeplab_768", "global_segformer_768", "local_boundary", "fullframe_combined_512")
METHODS = (*parent.METHODS, *NEW_METHODS)
sha256 = parent.sha256
read, save = parent.read, parent.save
csv_read, csv_write = parent.csv_read, parent.csv_write


def weight_method(method):
    return "local_combined" if method == "fullframe_combined_512" else method


def load_models(policy, store, method, device, loader=load_network):
    key = weight_method(method)
    family = REFINER_FAMILIES[key] if key.startswith("local_") else global_spec(key)[1]
    return [loader(run_path(store, key, f, 17) / "best.pt", policy["members"][key][f]["receipt"]["primary"], family, device) for f in range(5)]


def verify_inputs(policy, prepared, archive, store, previous, private768, private1024, images):
    prior = read(Path(previous) / "completed.json")
    if sha256(Path(previous) / "completed.json") != policy["parent_completion_sha256"] or prior["status"] != "source_external_complete":
        raise ValueError("Reviewed external completion changed")
    if prior["protocol_sha256"] != policy["parent_protocol_sha256"]:
        raise ValueError("Wrong parent external protocol")
    parent.verify_files(previous, prior["file_sha256"])
    print("Parent external arrays/receipts verified;no parent inference repeated", flush=True)
    parent_policy = read(Path(previous) / "analysis_protocol.json")
    if sha256(Path(previous) / "analysis_protocol.json") != policy["parent_protocol_sha256"] or policy["cases"] != parent_policy["cases"] or policy["geometry"] != parent_policy["geometry"]:
        raise ValueError("Source cohort/geometry changed")
    checked = parent.verify_inputs(parent_policy, prepared, archive, store)
    for method in ("global_deeplab_768", "global_segformer_768", "local_boundary"):
        for fold, member in enumerate(policy["members"][method]):
            run = run_path(store, method, fold, 17); expected = member["receipt"]["primary"]
            parent.verify_files(run, {"identity.json": member["identity_file_sha256"], "completed.json": member["completion_file_sha256"], "best.pt": expected["checkpoint_sha256"]})
            ident, done = read(run / "identity.json"), read(run / "completed.json")
            if ident != expected["identity"] or done["identity"] != ident or done["best_sha256"] != expected["checkpoint_sha256"] or member["fold"] != fold or member["seed"] != 17:
                raise ValueError("Additional frozen checkpoint differs")
            if method == "local_boundary" and member["receipt"]["coarse"] != policy["members"]["global_unet_768"][fold]["receipt"]["primary"]:
                raise ValueError("Boundary/coarse fold pairing differs")
    for key in parent.NETWORK_METHODS:
        if policy["members"][key] != parent_policy["members"][key]: raise ValueError("Parent weights changed")
    private_frames = {}
    for side, directory in ((768, private768), (1024, private1024)):
        if sha256(Path(directory) / "manifest.csv") != policy["private_manifest_sha256"][str(side)]:
            raise ValueError("Private prepared manifest changed")
        private_frames[side] = {r["patient_id"]: r for r in csv_read(Path(directory) / "manifest.csv")}
    selected = set()
    for case in policy["private_positive_controls"]:
        pid = case["patient_id"]; selected.add(pid)
        for side, directory in ((768, private768), (1024, private1024)):
            row = private_frames[side][pid]
            if row != case["rows"][str(side)] or row["split"] != "development" or int(row["fold"]) != case["fold"]:
                raise ValueError("Positive control is not matching-fold development")
            parent.verify_files(directory, {"cache/" + pid + ".npz": row["cache_sha256"]})
        parent.verify_files(images, {case["rows"]["768"]["image_relpath"]: case["rows"]["768"]["jpg_sha256"]})
        for method in ("global_unet_768", "global_unet_1024"):
            path = run_path(store, method, case["fold"], 17) / "val_metrics.csv"
            if sha256(path) != case["old_validation"][method]["file_sha256"]: raise ValueError("Old validation metrics changed")
    if len(selected) != len(policy["private_positive_controls"]): raise ValueError("Duplicate private control")
    return dict(**checked, parent_files_verified=len(prior["file_sha256"]), frozen_unique_checkpoints=40,
                private_development_controls=len(selected), private54_pixels_read=False)


@torch.inference_mode()
def infer(image, method, models, coarse, device, size, geometry):
    """Full RGB and optional frozen coarse cache only; no reference input."""
    if method not in NEW_METHODS: raise ValueError("Unexpected additional control")
    if method == "local_boundary":
        return parent.infer(image, method, models, coarse, device, size, geometry)
    # Each global baseline uses its recorded architecture resolution. The full
    # frame control uses the fine head at its original512 resolution without ROI.
    side = geometry.patch_side if method == "fullframe_combined_512" else global_spec(method)[2]
    cfg = replace(geometry, coarse_side=side)
    predictions = [predict_regions(image, model, None, device, cfg, "clean") for model in models]
    probabilities = [reconstruct(p, size, cfg).astype(np.float32) for p in predictions]
    return parent.average_five(probabilities), probabilities, predictions


def tensor_record(x, model):
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3 or x.dtype != torch.float32 or not torch.isfinite(x).all() or float(x.min()) < 0 or float(x.max()) > 1 or model.training:
        raise ValueError("Expected eval model with NCHW float32 RGB in[0,1]")
    if hasattr(model, "mean") and (not np.allclose(model.mean.flatten().cpu().numpy(), [.485, .456, .406], atol=1e-7, rtol=0) or not np.allclose(model.std.flatten().cpu().numpy(), [.229, .224, .225], atol=1e-7, rtol=0)):
        raise ValueError("Internal normalization differs from frozen model definition")
    return dict(shape=list(x.shape), dtype=str(x.dtype), minimum=float(x.min()), maximum=float(x.max()),
                channel_mean=x.mean((0, 2, 3)).cpu().tolist(), channel_sd=x.std((0, 2, 3), unbiased=False).cpu().tolist(),
                model_eval=not model.training, internal_mean=model.mean.flatten().cpu().tolist() if hasattr(model, "mean") else None,
                internal_std=model.std.flatten().cpu().tolist() if hasattr(model, "std") else None)


def match_probabilities(a, b, tolerance):
    if a.shape != b.shape: raise ValueError("Replay shape differs")
    error = float(np.max(np.abs(a - b)))
    masks = bool(np.array_equal(a >= .5, b >= .5))
    return dict(maximum_absolute_probability_difference=error, binary_masks_identical=masks,
                passed=error <= tolerance and masks)


@torch.inference_mode()
def diagnostics(policy, digest, prepared, archive, store, previous, private768, private1024, images, output, device, loader=load_network):
    directory = Path(output) / "diagnostics"; directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / "completed.json"
    if done_path.exists():
        done = read(done_path)
        if done["protocol_sha256"] != digest: raise ValueError("Diagnostic commitment changed")
        parent.verify_files(directory, done["file_sha256"]); return done
    geometry = CascadeConfig(**policy["geometry"]); tensors = []; private_rows = []; replay = []
    cases = {c["patient_id"]: c for c in policy["cases"]}
    # Fixed first two sorted validation IDs per fold, chosen without public scores.
    # Private54 is never used, and these checks are not a new performance cohort.
    for method in ("global_unet_768", "global_unet_1024"):
        models = load_models(policy, store, method, device, loader); side = global_spec(method)[2]
        directory_private = private768 if side == 768 else private1024
        for case in policy["private_positive_controls"]:
            row = case["rows"][str(side)]; model = models[case["fold"]]
            a = load_cached(directory_private, row); image = open_rgb(images, row)
            x, box = image_tensor(image, side, device)
            old_x = torch.from_numpy(a["image"].transpose(2, 0, 1).copy()).float()[None].to(device) / 255
            tensor_equal = bool(torch.equal(x, old_x)) and tuple(box) == tuple(a["box"])
            tensors.append(dict(patient_id=case["patient_id"], source="private_development", method=method, fold=case["fold"], **tensor_record(x, model)))
            prob = model(x)[0].sigmoid()[0, 0].float().cpu().numpy()
            ref = load_cached(private1024, case["rows"]["1024"]); gt = crop(ref["mask"] > 0, ref["box"])
            p = np.asarray(Image.fromarray(crop(prob, box)).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR))
            metrics = parent.compute_metrics(p >= .5, gt); old_dice = case["old_validation"][method]["dice"]
            delta = float(metrics["dice"] - old_dice)
            passed = tensor_equal and abs(delta) <= policy["private_dice_replay_tolerance"] and metrics["dice"] >= policy["private_positive_minimum_dice"] and not metrics["empty_prediction"]
            private_rows.append(dict(patient_id=case["patient_id"], method=method, fold=case["fold"], old_dice=old_dice,
                                     dice_difference=delta, RGB_tensor_exact_cache_match=tensor_equal, guard_passed=passed, **metrics))
        with zipfile.ZipFile(archive) as z:
            for pid in policy["public_replay_ids"]:
                case = cases[pid]; image = parent.original_rgb(z, case); predictions = []
                coarse = parent.coarse_predictions(Path(previous) / "methods/global_unet_768" / pid, image) if side == 768 else None
                for f, model in enumerate(models):
                    x, _ = image_tensor(image, side, device)
                    tensors.append(dict(patient_id=pid, source="public_original", method=method, fold=f, **tensor_record(x, model)))
                    pred = predict_regions(image, model, None, device, replace(geometry, coarse_side=side), "clean"); predictions.append(pred)
                    if side == 768:
                        replay.append(dict(patient_id=pid, method=method, fold=f, compared="native coarse content",
                                           **match_probabilities(pred["coarse_content"], coarse[f]["coarse_content"], policy["public_replay_tolerance"])))
                p = parent.average_five([reconstruct(r, case["reference_size"], replace(geometry, coarse_side=side)).astype(np.float32) for r in predictions])
                saved = parent.probability(Path(previous) / "methods" / method / pid / "probability.npy", tuple(reversed(case["reference_size"])))
                replay.append(dict(patient_id=pid, method=method, fold="ensemble", compared="source grid probability",
                                   **match_probabilities(p, saved, policy["public_replay_tolerance"])))
        del models; gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        print(f"Positive and replay checks complete:{method};0fits", flush=True)
    confidence = []
    for i, case in enumerate(policy["cases"], 1):
        for method in parent.METHODS:
            p = parent.probability(Path(previous) / "methods" / method / case["patient_id"] / "probability.npy", tuple(reversed(case["reference_size"])))
            quantiles = np.quantile(p, [.5, .95, .99, .999])
            confidence.append(dict(patient_id=case["patient_id"], method=method, country=case["country"],
                                   maximum=float(p.max()), mean=float(p.mean()), q50=float(quantiles[0]), q95=float(quantiles[1]),
                                   q99=float(quantiles[2]), q999=float(quantiles[3]),
                                   fraction_ge_fixed_0_5=float((p >= .5).mean()), interpretation="descriptive saved probability;not threshold search"))
        if i % 48 == 0: print(f"Saved probability audit:{i}/{len(policy['cases'])};0model calls", flush=True)
    csv_write(directory / "private_positive_controls.csv", private_rows)
    csv_write(directory / "public_probability_replay.csv", replay)
    csv_write(directory / "saved_probability_summaries.csv", confidence)
    save(directory / "input_tensor_records.json", tensors)
    passed = all(r["guard_passed"] for r in private_rows) and all(r["passed"] for r in replay)
    done = dict(protocol_sha256=digest, guard_passed=passed, private54_read=False, private_development_controls=len(policy["private_positive_controls"]),
                diagnostic_network_forward_calls=len(private_rows) + 10 * len(policy["public_replay_ids"]), public_replay_tolerance=policy["public_replay_tolerance"],
                interpretation="technical consistency/positive controls only;not causal scale test or new private evaluation",
                file_sha256={p.name: sha256(p) for p in directory.iterdir() if p.is_file() and p.name != "completed.json"})
    save(done_path, done); return done


def summaries(records, policy):
    ids = sorted(c["patient_id"] for c in policy["cases"]); lookup = {}
    for r in records:
        key = (r["method"], r["patient_id"])
        if key in lookup: raise ValueError("Duplicate patient-method")
        lookup[key] = r
    if set(lookup) != {(m, p) for m in METHODS for p in ids}: raise ValueError("Incomplete paired source cohort")
    groups = [("all", ids)]
    for key in ("country", "source_encoding"):
        for value in sorted({c[key] for c in policy["cases"]}): groups.append((key + ":" + value, sorted(c["patient_id"] for c in policy["cases"] if c[key] == value)))
    summary = []; events = []; contrasts = []
    for group, subset in groups:
        for method in METHODS:
            rows = [lookup[(method, p)] for p in subset]
            for metric in parent.METRIC_NAMES:
                v = np.array([r[metric] for r in rows], dtype=np.float64)
                row = dict(group=group, method=method, metric=metric, n_patients=len(subset), mean=float(v.mean()), median=float(np.median(v)), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.)
                if group == "all" and metric == "dice":
                    _, lo, hi = bootstrap_mean_ci(v, policy["bootstrap"], policy["statistics_seed"]); row.update(ci95_lower=lo, ci95_upper=hi)
                summary.append(row)
    vectors = {m: np.array([lookup[(m, p)]["dice"] for p in ids]) for m in METHODS}
    for baseline in METHODS:
        if baseline == "local_combined": continue
        a, b = vectors["local_combined"], vectors[baseline]
        contrasts.append(dict(contrast="combined_vs_control", proposed="local_combined", baseline=baseline,
                              **paired_bootstrap(a, b, policy["bootstrap"], policy["statistics_seed"]),
                              p_two_sided_signflip=sign_flip_pvalue(a - b, policy["permutations"], policy["statistics_seed"])))
    delta = vectors["local_combined"] - vectors["local_boundary"] - vectors["local_consistency"] + vectors["local_unet"]
    mean, lo, hi = bootstrap_mean_ci(delta, policy["bootstrap"], policy["statistics_seed"])
    contrasts.append(dict(contrast="factorial_interaction", proposed="C-B-K+F", baseline="zero", mean_delta_dice=mean, ci95_lower=lo, ci95_upper=hi,
                          n_patients=len(ids), p_two_sided_signflip=sign_flip_pvalue(delta, policy["permutations"], policy["statistics_seed"])))
    for r, p in zip(contrasts, holm_adjust([r["p_two_sided_signflip"] for r in contrasts])):
        r.update(holm_p_across10=float(p), interpretation="post_result_exploratory;conditional_on_fitted_weights;no training replicate uncertainty;no causal interaction claim")
    for method in METHODS:
        rows = [lookup[(method, p)] for p in ids]
        for event, n in (("empty_prediction", sum(int(r["empty_prediction"]) for r in rows)), ("dice_below_0_5", sum(r["dice"] < .5 for r in rows))):
            rate, lo, hi = wilson_rate_ci(n, len(ids)); events.append(dict(method=method, event=event, events=n, n_patients=len(ids), rate=rate, wilson_ci95_lower=lo, wilson_ci95_upper=hi))
    return summary, contrasts, events


def qc_sheets(output, prepared, previous, cases):
    qc = Path(output) / "qc"; qc.mkdir(exist_ok=True); manifest = []
    display = ("local_combined", *NEW_METHODS)
    labels = ("Combined", "DeepLab768", "SegFormer768", "Boundary", "Fine fullframe512")
    for start in range(0, len(cases), 12):
        canvas = Image.new("RGB", (1560, 1176), "white"); subset = cases[start:start + 12]
        for i, case in enumerate(subset):
            with Image.open(parent.contained(prepared, case["image_path"])) as im: rgb = np.array(im.convert("RGB"))
            gt = parent.reference_mask(prepared, case); yy, xx = np.nonzero(gt)
            bounds = (max(0, int(xx.min()) - 24), max(0, int(yy.min()) - 24), min(gt.shape[1], int(xx.max()) + 25), min(gt.shape[0], int(yy.max()) + 25))
            tile = Image.new("RGB", (780, 196), "white"); draw = ImageDraw.Draw(tile); draw.text((2, 0), case["patient_id"] + " | GT crop DISPLAY ONLY", fill="black")
            ref = rgb.copy(); ref[gt] = (.65 * rgb[gt] + .35 * np.array([0, 220, 80])).astype(np.uint8); overlays = [("Reference", ref)]
            for method, label in zip(display, labels):
                root = previous if method in parent.METHODS else output
                p = parent.probability(Path(root) / "methods" / method / case["patient_id"] / "probability.npy", gt.shape) >= .5
                a = rgb.copy(); a[p] = (.65 * rgb[p] + .35 * np.array([0, 100, 255])).astype(np.uint8); a[gt & ~ndi.binary_erosion(gt)] = (255, 220, 0)
                overlays.append((label, a))
            for j, (label, a) in enumerate(overlays):
                patch = Image.fromarray(a).crop(bounds); patch.thumbnail((128, 160), Image.Resampling.LANCZOS)
                draw.text((j * 130 + 1, 17), label, fill="black"); tile.paste(patch, (j * 130, 32))
            canvas.paste(tile, ((i % 2) * 780, (i // 2) * 196))
        name = f"qc/sheet_{start // 12 + 1:02d}.png"; canvas.save(Path(output) / name)
        manifest.append(dict(file=name, patient_ids=[c["patient_id"] for c in subset], GT_crop_display_only=True))
    save(Path(output) / "qc_manifest.json", manifest)


def collect_return(output):
    output = Path(output)
    files = [p for p in output.rglob("*") if p.is_file() and (p.suffix in (".csv", ".json", ".md") or p.parent.name == "qc")]
    with zipfile.ZipFile(output / "public_external_controls_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files): z.write(p, p.relative_to(output).as_posix())


def finish(output, digest, status, details):
    output = Path(output); save(output / "summary.json", dict(status=status, **details))
    files = [p for p in output.rglob("*") if p.is_file() and p.suffix != ".zip" and p != output / "completed.json"]
    returned = [p for p in files if p.suffix in (".csv", ".json", ".md") or p.parent.name == "qc"]
    done = dict(status=status, protocol_sha256=digest, completed_utc=datetime.now(timezone.utc).isoformat(), training_fits=0,
                file_sha256={p.relative_to(output).as_posix(): sha256(p) for p in sorted(files)},
                returned_file_sha256={p.relative_to(output).as_posix(): sha256(p) for p in sorted(returned)},
                arrays_kept_server_only=True, original54_recomputed=False)
    save(output / "completed.json", done); collect_return(output); return done


def evaluate(policy, digest, prepared, archive, store, previous, private768, private1024, images, output, device, loader=load_network):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        done = read(output / "completed.json")
        if done["protocol_sha256"] != digest: raise ValueError("Completed control protocol changed")
        parent.verify_files(output, done["file_sha256"]); collect_return(output); return done
    try:
        check = diagnostics(policy, digest, prepared, archive, store, previous, private768, private1024, images, output, device, loader)
    except Exception as error:
        save(output / "diagnostic_exception.json", dict(error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc(), controls_stopped=True))
        print(f"Diagnostic exception:{type(error).__name__}:{error};controls STOPPED;return diagnostic ZIP", flush=True)
        return finish(output, digest, "diagnostic_guard_failed", dict(training_fits=0, additional_controls_executed=False, diagnostic_exception=True))
    if not check["guard_passed"]:
        print("Technical guard failed;additional public controls STOPPED;return diagnostics;no retuning", flush=True)
        return finish(output, digest, "diagnostic_guard_failed", dict(training_fits=0, additional_controls_executed=False, diagnostic=check))
    geometry = CascadeConfig(**policy["geometry"]); all_done = []
    for method in parent.METHODS:
        for case in policy["cases"]:
            all_done.append(parent.case_done(Path(previous) / "methods" / method / case["patient_id"], policy["parent_protocol_sha256"], method, case))
    with zipfile.ZipFile(archive) as z:
        for method in NEW_METHODS:
            pending = [c for c in policy["cases"] if not parent.case_done(output / "methods" / method / c["patient_id"], digest, method, c)]
            models = load_models(policy, store, method, device, loader) if pending else []
            for i, case in enumerate(policy["cases"], 1):
                dest = output / "methods" / method / case["patient_id"]; done = parent.case_done(dest, digest, method, case)
                if done: all_done.append(done); continue
                start = time.perf_counter(); image = parent.original_rgb(z, case)
                coarse = parent.coarse_predictions(Path(previous) / "methods/global_unet_768" / case["patient_id"], image) if method == "local_boundary" else None
                p, member_p, predictions = infer(image, method, models, coarse, device, case["reference_size"], geometry)
                gt = parent.reference_mask(prepared, case)
                done = parent.store_case(dest, p, member_p, predictions, gt, case, method, digest, time.perf_counter() - start); all_done.append(done)
                if i % 12 == 0 or i == len(policy["cases"]): print(f"Additional public {method}:{i}/{len(policy['cases'])};0fits", flush=True)
            del models; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
            collect_return(output)
    rows = [d["metrics"] for d in all_done]; members = [r for d in all_done for r in d["member_metrics"]]
    summary, contrasts, events = summaries(rows, policy)
    for name, values in (("patient_metrics.csv", rows), ("member_metrics.csv", members), ("summary_metrics.csv", summary), ("paired_dice_contrasts.csv", contrasts), ("failure_events.csv", events)):
        csv_write(output / name, values)
    qc_sheets(output, prepared, previous, policy["cases"])
    (output / "README.md").write_text("Post-result exploratory controls;0training fits. Parent6outputs reused unchanged;4additional outputs use existing weights. Fullframe_combined_512 is direct fine model on full RGB,without coarse proposals or GT crops. Global DeepLab/SegFormer use768;fixed0.5 threshold. Source-provided joint union is scoring only;all212retained. Private controls are fixed10development patients solely for technical positive/replay checks;no54read. New10comparison family comprises9combined-versus-control contrasts and1C-B-K+F descriptive interaction;original Holm5 results unchanged. Conditional patient uncertainty excludes training variation and cannot establish causal scale effects,loss synergy,or innovation. Cached timings not standalone latency. Array/mask outputs remain server-only.\n", encoding="utf8")
    return finish(output, digest, "source_external_controls_complete", dict(training_fits=0, n_patients=len(policy["cases"]), methods=list(METHODS),
                  new_methods=list(NEW_METHODS), reused_parent_outputs=len(parent.METHODS), patient_method_rows=len(rows), member_rows=len(members),
                  new_control_network_forward_calls=sum(d["network_forward_calls"] for d in all_done if d["method"] in NEW_METHODS),
                  diagnostic_network_forward_calls=check["diagnostic_network_forward_calls"], diagnostic_guard_passed=True,
                  public_fitting=False, threshold_search=False, GT_crop_inference=False, original54_recomputed=False,
                  novelty_or_causal_scale_claim=False, comparisons_post_result_exploratory=True))


def main():
    ap = argparse.ArgumentParser(description="Existing-environment zero-fit source-external controls and technical diagnostics")
    ap.add_argument("--archive", default="public_datasets/eyes_defy_anemia.zip"); ap.add_argument("--prepared-public", default="results_public_reference_prepare_v1")
    ap.add_argument("--model-store", default="results_round4"); ap.add_argument("--previous", default="results_public_external_v1")
    ap.add_argument("--prepared", default="prepared"); ap.add_argument("--prepared-1024", default="prepared_round2_1024"); ap.add_argument("--images", default="images")
    ap.add_argument("--output", default="results_public_external_controls_v1"); ap.add_argument("--protocol", required=True); ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--cpu-threads", type=int, default=4); args = ap.parse_args(); spec = Path(args.protocol).resolve(); digest = sha256(spec)
    if digest != args.protocol_sha256: raise ValueError("Frozen controls protocol changed")
    policy = read(spec); parent.verify_files(Path(__file__).resolve().parents[2], policy["source_sha256"])
    if policy["schema"] != "conjseg_public_external_controls_v1" or policy["methods"] != list(METHODS) or len(policy["cases"]) != 212 or policy["training_fits"] != 0 or policy["threshold"] != .5:
        raise ValueError("Wrong fixed additional control plan")
    output = Path(args.output).resolve()
    for p in (args.prepared_public, args.archive, args.model_store, args.previous, args.prepared, args.prepared_1024, args.images, spec):
        p = Path(p).resolve()
        if output == p or output.is_relative_to(p) or p.is_relative_to(output): raise ValueError("Output overlaps protected source")
    if output.exists() and any(output.iterdir()) and not (output / "analysis_protocol.json").exists(): raise ValueError("Unrelated nonempty output")
    if args.cpu_threads < 1: raise ValueError("Positive CPU thread count required")
    checked = verify_inputs(policy, args.prepared_public, args.archive, args.model_store, args.previous, args.prepared, args.prepared_1024, args.images)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA not visible in selected existing Python;no installation performed")
    output.mkdir(parents=True, exist_ok=True); frozen = output / "analysis_protocol.json"
    if frozen.exists() and frozen.read_bytes() != spec.read_bytes(): raise ValueError("Partial protocol changed")
    frozen.write_bytes(spec.read_bytes())
    if not (output / "completed.json").exists():
        save(output / "input_verification.json", checked); torch.set_num_threads(args.cpu_threads); seed_all(17)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        environment = actual_environment(); epath = output / "environment.json"
        if epath.exists() and read(epath) != environment: raise ValueError("Partial environment changed")
        save(epath, environment)
    done = evaluate(policy, digest, args.prepared_public, args.archive, args.model_store, args.previous, args.prepared, args.prepared_1024, args.images, output, torch.device("cuda"))
    print(f"{done['status']};return {output / 'public_external_controls_results.zip'};0fits", flush=True)


if __name__ == "__main__": main()
