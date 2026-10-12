"""Frozen private-trained ensembles -> source-provided public union references.

Inference only. Full EXIF-oriented ORIGINAL RGB is the input. Supplied cutouts,
masks and GT crops never enter the inference path. No target adaptation/tuning.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
import torch

from .audit import sha256
from .cascade import CascadeConfig, predict_regions, reconstruct
from .holdout import average_five, refine_cached
from .metrics import compute_metrics, METRIC_NAMES
from .public_reference_prepare_v1 import csv_read, csv_write
from .round3 import REFINER_FAMILIES
from .round4 import load_network, run_path, global_spec, actual_environment
from .stats import bootstrap_mean_ci, paired_bootstrap, sign_flip_pvalue, holm_adjust, wilson_rate_ci
from .training import seed_all

NETWORK_METHODS = ("global_unet_768", "global_unet_1024", "local_unet", "local_consistency", "local_combined")
AVERAGE = "fusion_average"
METHODS = (*NETWORK_METHODS, AVERAGE)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def contained(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or Path(name).is_absolute():
        raise ValueError("Invalid relative receipt path")
    p = (root / name).resolve()
    if ".." in Path(name).parts or not p.is_relative_to(root):
        raise ValueError("Receipt path escapes its root")
    return p


def verify_files(root, hashes):
    for n, h in hashes.items():
        if sha256(contained(root, n)) != h:
            raise ValueError("Receipt file changed: " + n)


def verify_inputs(policy, prepared, archive, store):
    done = read(Path(prepared) / "completed.json")
    if sha256(Path(prepared) / "completed.json") != policy["preparation_completion_sha256"]:
        raise ValueError("Reviewed preparation completion changed; preserve the returned preparation")
    if done["status"] != "candidates_prepared" or done["protocol_sha256"] != policy["preparation_protocol_sha256"]:
        raise ValueError("Wrong preparation source")
    verify_files(prepared, done["file_sha256"])
    assets = csv_read(Path(prepared) / "server_asset_inventory.csv")
    asset_hashes = {}
    for r in assets:
        if r["path"] in asset_hashes and asset_hashes[r["path"]] != r["sha256"]:
            raise ValueError("Conflicting duplicate asset identity")
        asset_hashes[r["path"]] = r["sha256"]
    verify_files(prepared, asset_hashes)
    if sha256(archive) != policy["public_archive_sha256"]:
        raise ValueError("Reviewed public archive changed")
    manifest = {r["patient_key"]: r for r in csv_read(Path(prepared) / "patient_manifest.csv")}
    selected = {p for p, r in manifest.items() if r["primary_union_candidate_available"] == "True"}
    if selected != {r["patient_id"] for r in policy["cases"]}:
        raise ValueError("Public union cohort changed")
    for case in policy["cases"]:
        r = manifest[case["patient_id"]]
        if json.loads(r["source_original"]) != [case["original_path"]]:
            raise ValueError("Original source identity differs")
        for k in ("mask_path", "mask_file_sha256", "binary_mask_sha256", "image_path", "image_file_sha256"):
            if r[k] != case[k]:
                raise ValueError("Prepared case identity differs")
        if [int(r["width"]), int(r["height"])] != case["reference_size"]:
            raise ValueError("Reference canvas differs")
    for method in NETWORK_METHODS:
        for fold, member in enumerate(policy["members"][method]):
            run = run_path(store, method, fold, 17)
            verify_files(run, {"identity.json": member["identity_file_sha256"], "completed.json": member["completion_file_sha256"],
                               "best.pt": member["receipt"]["primary"]["checkpoint_sha256"]})
            ident = read(run / "identity.json"); completion = read(run / "completed.json")
            expected = member["receipt"]["primary"]
            if ident != expected["identity"] or completion["identity"] != ident or completion["best_sha256"] != expected["checkpoint_sha256"]:
                raise ValueError("Frozen checkpoint identity differs")
            if member["fold"] != fold or member["seed"] != 17:
                raise ValueError("Wrong frozen member order")
            if method.startswith("local_") and member["receipt"]["coarse"] != policy["members"]["global_unet_768"][fold]["receipt"]["primary"]:
                raise ValueError("Local/coarse fold pairing differs")
    return dict(output_hashes=len(done["file_sha256"]), asset_records=len(assets), unique_assets=len(asset_hashes),
                frozen_checkpoints=5 * len(NETWORK_METHODS), public_cases=len(selected), private_patient_pixels_read=False)


def original_rgb(z, case):
    raw = z.read(case["original_path"])
    if hashlib.sha256(raw).hexdigest() != case["original_sha256"]:
        raise ValueError("Original RGB bytes differ")
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        result = ImageOps.exif_transpose(im).convert("RGB")
    return result


def reference_mask(prepared, case):
    path = contained(prepared, case["mask_path"])
    if sha256(path) != case["mask_file_sha256"]:
        raise ValueError("Source reference PNG changed")
    with Image.open(path) as im:
        a = np.array(im)
    if a.ndim != 2 or (a.shape[1], a.shape[0]) != tuple(case["reference_size"]) or not np.isin(a, [0, 255]).all():
        raise ValueError("Invalid source reference mask")
    gt = a > 0
    if not gt.any() or hashlib.sha256(gt.astype(np.uint8).tobytes()).hexdigest() != case["binary_mask_sha256"]:
        raise ValueError("Source reference support differs")
    return gt


def probability(path, shape):
    p = np.load(path, allow_pickle=False)
    if p.dtype != np.float32 or p.shape != tuple(shape) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid saved probability")
    return p


def case_done(directory, digest, method, case):
    path = Path(directory) / "completed.json"
    if not path.exists():
        return None
    done = read(path)
    if done["protocol_sha256"] != digest or done["method"] != method or done["patient_id"] != case["patient_id"]:
        raise ValueError("Completed case commitment changed")
    verify_files(directory, done["file_sha256"])
    probability(Path(directory) / "probability.npy", tuple(reversed(case["reference_size"])))
    return done


def coarse_predictions(directory, image):
    with np.load(Path(directory) / "coarse.npz", allow_pickle=False) as data:
        content, native = data["content"], data["native_size"]
    if content.dtype != np.float32 or content.ndim != 3 or content.shape[0] != 5 or tuple(native) != image.size:
        raise ValueError("Cached coarse geometry differs from full original")
    if not np.isfinite(content).all() or (content < 0).any() or (content > 1).any():
        raise ValueError("Invalid coarse cache probabilities")
    return [dict(coarse_content=p, native_size=image.size, boxes=[], patch_probabilities=[]) for p in content]


@torch.inference_mode()
def infer(image, method, models, coarse, device, size, geometry):
    """RGB-only function. Its signature contains no reference/mask/case record."""
    predictions = []
    for fold in range(5):
        if method.startswith("local_"):
            predictions.append(refine_cached(image, coarse[fold], models[fold], device, geometry, "clean"))
        else:
            cfg = geometry if method == "global_unet_768" else replace(geometry, coarse_side=1024)
            predictions.append(predict_regions(image, models[fold], None, device, cfg, "clean"))
    cfg = geometry if method != "global_unet_1024" else replace(geometry, coarse_side=1024)
    probabilities = [reconstruct(p, size, cfg).astype(np.float32) for p in predictions]
    return average_five(probabilities), probabilities, predictions


def metrics_row(p, gt, case, method, fold=None):
    pred = p >= .5
    result = compute_metrics(pred, gt)
    result.update(patient_id=case["patient_id"], method=method, country=case["country"],
                  source_encoding=case["source_encoding"], reference_grid="source_union_800x1067",
                  threshold=.5, tp_pixels=int((pred & gt).sum()), fp_pixels=int((pred & ~gt).sum()), fn_pixels=int((~pred & gt).sum()),
                  source_reference="provided_joint_union;fixed_source_support;not_independent_reannotation")
    if fold is not None:
        result["fold"] = fold
    return result


def store_case(directory, p, member_p, predictions, gt, case, method, digest, elapsed):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "probability.npy", p.astype(np.float32), allow_pickle=False)
    Image.fromarray((p >= .5).astype(np.uint8) * 255).save(directory / "prediction.png")
    files = ["probability.npy", "prediction.png"]
    if method == "global_unet_768":
        np.savez_compressed(directory / "coarse.npz", content=np.stack([r["coarse_content"] for r in predictions]).astype(np.float32),
                            native_size=np.array(predictions[0]["native_size"], dtype=np.int64))
        files.append("coarse.npz")
    row = metrics_row(p, gt, case, method)
    members = [metrics_row(q, gt, case, method, f) for f, q in enumerate(member_p)]
    forward_calls = 0 if method == AVERAGE else sum(len(r["patch_probabilities"]) if method.startswith("local_") else 1 for r in predictions)
    done = dict(protocol_sha256=digest, patient_id=case["patient_id"], method=method, metrics=row, member_metrics=members,
                file_sha256={n: sha256(directory / n) for n in files}, pipeline_predictions=0 if method == AVERAGE else 5,
                network_forward_calls=forward_calls, cached_inference_seconds=elapsed,
                timing_comparable_across_methods=False, reference_read_after_RGB_inference=True,
                original_input_size=list(predictions[0]["native_size"]) if predictions else None,
                proposals=[dict(fold=f, boxes=r["boxes"], raw_component_count=r.get("raw_component_count"), proposal_mode=r.get("proposal_mode"))
                           for f, r in enumerate(predictions)] if method.startswith("local_") else [])
    save(directory / "completed.json", done)
    return done


def summaries(records, policy):
    expected = {c["patient_id"] for c in policy["cases"]}
    lookup = {}
    for r in records:
        k = (r["method"], r["patient_id"])
        if k in lookup:
            raise ValueError("Duplicate patient/method scores")
        lookup[k] = r
    if set(lookup) != {(m, p) for m in METHODS for p in expected}:
        raise ValueError("Incomplete paired public cohort")
    groups = [("all", sorted(expected))]
    for country in ("India", "Italy"):
        ids = sorted(c["patient_id"] for c in policy["cases"] if c["country"].lower() == country.lower())
        if ids: groups.append((country, ids))
    for encoding in ("alpha_ge128", "rgb_not_exact_white"):
        ids = sorted(c["patient_id"] for c in policy["cases"] if c["source_encoding"] == encoding)
        if ids: groups.append((encoding, ids))
    summary = []
    for group, ids in groups:
        for method in METHODS:
            rows = [lookup[(method, p)] for p in ids]
            for metric in METRIC_NAMES:
                values = np.array([r[metric] for r in rows], dtype=np.float64)
                interval = bootstrap_mean_ci(values, policy["bootstrap"], policy["statistics_seed"]) if group == "all" and metric == "dice" else None
                summary.append(dict(group=group, method=method, metric=metric, n_patients=len(ids), mean=float(values.mean()),
                                    median=float(np.median(values)), sd=float(values.std(ddof=1)) if len(ids) > 1 else 0.,
                                    **({"ci95_lower": interval[1], "ci95_upper": interval[2]} if interval else {})))
    ids = sorted(expected); proposed = np.array([lookup[("local_combined", p)]["dice"] for p in ids])
    contrasts = []
    for baseline in METHODS:
        if baseline == "local_combined": continue
        values = np.array([lookup[(baseline, p)]["dice"] for p in ids])
        contrasts.append(dict(proposed="local_combined", baseline=baseline,
                              **paired_bootstrap(proposed, values, policy["bootstrap"], policy["statistics_seed"]),
                              p_two_sided_signflip=sign_flip_pvalue(proposed - values, policy["permutations"], policy["statistics_seed"]),
                              interpretation="exploratory_source_external;conditional_on_existing_ensembles"))
    for row, p in zip(contrasts, holm_adjust([r["p_two_sided_signflip"] for r in contrasts])):
        row["holm_p_across5"] = float(p)
    events = []
    for method in METHODS:
        rows = [lookup[(method, p)] for p in ids]
        for name, n in (("empty_prediction", sum(r["empty_prediction"] for r in rows)), ("dice_below_0_5", sum(r["dice"] < .5 for r in rows))):
            rate, lo, hi = wilson_rate_ci(n, len(ids))
            events.append(dict(method=method, event=name, events=n, n_patients=len(ids), rate=rate, wilson_ci95_lower=lo, wilson_ci95_upper=hi))
    return summary, contrasts, events


def qc_sheets(output, prepared, cases):
    qc = Path(output) / "qc"; qc.mkdir(exist_ok=True); manifest = []
    methods = ("global_unet_1024", "local_combined", AVERAGE)
    for start in range(0, len(cases), 12):
        canvas = Image.new("RGB", (1320, 1128), "white")
        subset = cases[start:start + 12]
        for i, case in enumerate(subset):
            with Image.open(contained(prepared, case["image_path"])) as im: rgb = np.array(im.convert("RGB"))
            gt = reference_mask(prepared, case); yy, xx = np.nonzero(gt)
            box = (max(0, int(xx.min()) - 24), max(0, int(yy.min()) - 24), min(gt.shape[1], int(xx.max()) + 25), min(gt.shape[0], int(yy.max()) + 25))
            tile = Image.new("RGB", (660, 188), "white"); d = ImageDraw.Draw(tile)
            d.text((2, 0), case["patient_id"] + " | source reference; model QC", fill="black")
            overlays = []
            ref = rgb.copy(); ref[gt] = (.65 * rgb[gt] + .35 * np.array([0, 220, 80])).astype(np.uint8)
            overlays.append(("Reference", ref))
            for method in methods:
                p = probability(Path(output) / "methods" / method / case["patient_id"] / "probability.npy", gt.shape) >= .5
                a = rgb.copy(); a[p] = (.65 * rgb[p] + .35 * np.array([0, 100, 255])).astype(np.uint8)
                edge = gt & ~ndi.binary_erosion(gt); a[edge] = (255, 220, 0)
                overlays.append((method.replace("global_unet_", "U").replace("local_", ""), a))
            for j, (label, a) in enumerate(overlays):
                patch = Image.fromarray(a).crop(box); patch.thumbnail((163, 160), Image.Resampling.LANCZOS)
                d.text((j * 165 + 2, 17), label, fill="black"); tile.paste(patch, (j * 165, 32))
            canvas.paste(tile, ((i % 2) * 660, (i // 2) * 188))
        name = f"qc/sheet_{start // 12 + 1:02d}.png"; canvas.save(Path(output) / name)
        manifest.append(dict(file=name, patient_ids=[c["patient_id"] for c in subset], GT_crop_display_only=True))
    save(Path(output) / "qc_manifest.json", manifest)


def collect_return(output):
    output = Path(output)
    files = [p for p in output.rglob("*") if p.is_file() and (p.suffix in (".csv", ".json", ".md") or p.parent.name == "qc")]
    with zipfile.ZipFile(output / "public_external_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files): z.write(p, p.relative_to(output).as_posix())


def evaluate(policy, digest, prepared, archive, store, output, device, loader=load_network):
    """Sequential frozen methods; resume only verified complete patient cases."""
    output = Path(output); geometry = CascadeConfig(**policy["geometry"]); cases = policy["cases"]
    output.mkdir(parents=True, exist_ok=True)
    previous = output / "analysis_protocol.json"
    if previous.exists() and sha256(previous) != digest: raise ValueError("Partial output protocol changed")
    if (output / "completed.json").exists():
        done = read(output / "completed.json")
        if done["protocol_sha256"] != digest: raise ValueError("Completed protocol changed")
        verify_files(output, done["file_sha256"])
        for method in METHODS:
            for case in cases: case_done(output / "methods" / method / case["patient_id"], digest, method, case)
        collect_return(output); print("Verified completed external evaluation reused;0newfits;0newinferences", flush=True)
        return done
    all_done = []
    with zipfile.ZipFile(archive) as z:
        for method in METHODS:
            pending = [c for c in cases if not case_done(output / "methods" / method / c["patient_id"], digest, method, c)]
            models = []
            if pending and method != AVERAGE:
                family = REFINER_FAMILIES[method] if method.startswith("local_") else global_spec(method)[1]
                models = [loader(run_path(store, method, f, 17) / "best.pt", policy["members"][method][f]["receipt"]["primary"], family, device) for f in range(5)]
            for i, case in enumerate(cases, 1):
                dest = output / "methods" / method / case["patient_id"]
                done = case_done(dest, digest, method, case)
                if done: all_done.append(done); continue
                start = time.perf_counter()
                if method == AVERAGE:
                    shape = tuple(reversed(case["reference_size"]))
                    a = probability(output / "methods/global_unet_768" / case["patient_id"] / "probability.npy", shape)
                    b = probability(output / "methods/local_combined" / case["patient_id"] / "probability.npy", shape)
                    p, member_p, predictions = (a + b) * np.float32(.5), [], []
                else:
                    image = original_rgb(z, case)
                    coarse = coarse_predictions(output / "methods/global_unet_768" / case["patient_id"], image) if method.startswith("local_") else None
                    p, member_p, predictions = infer(image, method, models, coarse, device, case["reference_size"], geometry)
                elapsed = time.perf_counter() - start
                gt = reference_mask(prepared, case)  # after prediction; source support is scoring only
                done = store_case(dest, p, member_p, predictions, gt, case, method, digest, elapsed)
                all_done.append(done)
                if i % 12 == 0 or i == len(cases): print(f"Public external {method}:{i}/{len(cases)};0fits;frozen weights", flush=True)
            del models; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
            collect_return(output)
    rows = [d["metrics"] for d in all_done]; members = [r for d in all_done for r in d["member_metrics"]]
    summary, contrasts, events = summaries(rows, policy)
    for name, data in (("patient_metrics.csv", rows), ("member_metrics.csv", members), ("summary_metrics.csv", summary), ("paired_dice_contrasts.csv", contrasts), ("failure_events.csv", events)):
        csv_write(output / name, data)
    qc_sheets(output, prepared, cases)
    save(output / "summary.json", dict(status="source_external_complete", n_patients=len(cases), methods=list(METHODS), training_fits=0,
         pipeline_predictions=sum(d["pipeline_predictions"] for d in all_done), network_forward_calls=sum(d["network_forward_calls"] for d in all_done),
         original_RGB_input=True, GT_crop_inference=False, adaptation=False, parameter_or_threshold_search=False,
         reference="fixed source-provided joint union;technical QC completed;no new ophthalmologist adjudication",
         source_exception_cases_not_scored=policy["source_exceptions"], india2_color_anomaly_retained=True,
         original54_recomputed=False, private_patient_pixels_read=False,
         average_fusion="0.5*(fivefold_coarse_mean+fivefold_combined_mean);no separately trained model or extra network forwards",
         member_rows=sum(len(d["member_metrics"]) for d in all_done), average_member_metrics_not_computed=True,
         uncertainty="paired patient bootstrap conditional on fitted ensembles;Holm5 exploratory contrasts;countries/encoding descriptive",
         public_future_role="If later used for development, this dataset is no longer an untouched external test for that later work"))
    (output / "README.md").write_text("Frozen private-trained ensembles evaluated on public source union references. Full EXIF-oriented original RGB input;fixed threshold0.5 and original geometry;no training/adaptation/GT inference crops. All212source-union patients retained including India2;6author exceptions not substituted. Reference is provided joint source support,not OR of component masks or new clinical adjudication. One ensemble row/patient/method;folds are not patients. Averaging is fixed0.5,no new model. All probabilities,prediction PNGs and coarse caches stay server-only. Cached runtime is not a standalone latency comparison. If public data later becomes development data,disclose that role change. Original54internal analysis untouched. Mirror rights declaration does not establish independently verified original rights.\n", encoding="utf8")
    files = [p for p in output.rglob("*") if p.is_file() and p.suffix != ".zip" and p.name != "completed.json"]
    files += list((output / "methods").glob("*/*/completed.json"))
    returned_files = [p for p in files if p.suffix in (".csv", ".json", ".md") or p.parent.name == "qc"]
    done = dict(status="source_external_complete", protocol_sha256=digest, completed_utc=datetime.now(timezone.utc).isoformat(),
                training_fits=0, n_patients=len(cases), file_sha256={p.relative_to(output).as_posix(): sha256(p) for p in sorted(files)},
                returned_file_sha256={p.relative_to(output).as_posix(): sha256(p) for p in sorted(returned_files)},
                arrays_kept_server_only=True, original54_recomputed=False, private_patient_pixels_read=False)
    save(output / "completed.json", done); collect_return(output)
    print(f"Completed public source external evaluation:{len(cases)}patients;0fits;return {output / 'public_external_results.zip'}", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser(description="Zero-training frozen private->public segmentation validation;existing environment")
    ap.add_argument("--archive", default="public_datasets/eyes_defy_anemia.zip")
    ap.add_argument("--prepared-public", default="results_public_reference_prepare_v1")
    ap.add_argument("--model-store", default="results_round4")
    ap.add_argument("--output", default="results_public_external_v1")
    ap.add_argument("--protocol", required=True); ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--cpu-threads", type=int, default=4)
    args = ap.parse_args(); spec = Path(args.protocol).resolve(); digest = sha256(spec)
    if digest != args.protocol_sha256: raise ValueError("Frozen external protocol changed")
    policy = read(spec); base = Path(__file__).resolve().parents[2]
    verify_files(base, policy["source_sha256"])
    if policy["schema"] != "conjseg_public_external_v1" or policy["methods"] != list(METHODS) or len(policy["cases"]) != 212 or policy["threshold"] != .5 or policy["training_fits"] != 0:
        raise ValueError("Wrong fixed public evaluation plan")
    output = Path(args.output).resolve()
    for p in (Path(args.prepared_public).resolve(), Path(args.archive).resolve(), Path(args.model_store).resolve(), spec):
        if output == p or output.is_relative_to(p) or p.is_relative_to(output): raise ValueError("Output overlaps protected source")
    if output.exists() and any(output.iterdir()) and not (output / "analysis_protocol.json").exists(): raise ValueError("Unrelated nonempty output")
    if args.cpu_threads < 1: raise ValueError("Positive CPU thread count required")
    verification = verify_inputs(policy, args.prepared_public, args.archive, args.model_store)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA not visible in selected existing Python;no installation performed")
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "analysis_protocol.json"
    if frozen.exists() and frozen.read_bytes() != spec.read_bytes(): raise ValueError("Previous frozen plan differs")
    frozen.write_bytes(spec.read_bytes())
    if (output / "completed.json").exists():
        evaluate(policy, digest, args.prepared_public, args.archive, args.model_store, output, torch.device("cuda"))
        return
    save(output / "input_verification.json", verification)
    torch.set_num_threads(args.cpu_threads); seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    environment = actual_environment(); environment_path = output / "environment.json"
    if environment_path.exists() and read(environment_path) != environment: raise ValueError("Partial run environment changed")
    save(environment_path, environment)
    evaluate(policy, digest, args.prepared_public, args.archive, args.model_store, output, torch.device("cuda"))


if __name__ == "__main__":
    main()
