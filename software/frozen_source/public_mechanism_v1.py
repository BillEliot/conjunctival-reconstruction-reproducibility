"""Bounded post-result inference ablations; zero training and no private pixels.

Same512 input for the existing four fine models and one whole-image-trained
U-Net. A single-fullframe reconstruction intervention isolates old tapering.
All weights, proposals, source reference, threshold and parent results are fixed.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import torch

from . import public_external_v1 as p
from . import public_external_controls_v1 as c
from .cascade import CascadeConfig, predict_regions, reconstruct, resize_probability
from .round3 import REFINER_FAMILIES
from .round4 import global_spec, load_network, run_path, actual_environment
from .training import seed_all

WEIGHTS = {"fullframe_plain_512": "local_unet", "fullframe_boundary_512": "local_boundary",
           "fullframe_consistency_512": "local_consistency", "fullframe_globalunet_512": "global_unet_768",
           "cascade_fullframe_identity": "local_combined"}
NEW_METHODS = tuple(WEIGHTS)
METHODS = (*c.METHODS, *NEW_METHODS)
DIRECT = "fullframe_combined_512"
IDENTITY = "cascade_fullframe_identity"
save, read, sha256, csv_read, csv_write = p.save, p.read, p.sha256, p.csv_read, p.csv_write


def single_fullframe(prediction):
    w, h = prediction["native_size"]
    return len(prediction["boxes"]) == 1 and tuple(prediction["boxes"][0]) == (0, 0, w, h)


def reconstruct_identity(prediction, output_size, geometry):
    """Only one exact whole-image ROI bypasses taper; no semantic/score gate."""
    if single_fullframe(prediction):
        if len(prediction["patch_probabilities"]) != 1: raise ValueError("Fullframe patch support missing")
        q = prediction["patch_probabilities"][0]
        if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any(): raise ValueError("Invalid fine probability")
        return resize_probability(q, output_size).astype(np.float32)
    return reconstruct(prediction, output_size, geometry).astype(np.float32)


@torch.inference_mode()
def infer(image, method, models, coarse, device, size, geometry):
    if method not in NEW_METHODS: raise ValueError("Unexpected fixed method")
    if method == IDENTITY:
        predictions = [p.refine_cached(image, coarse[f], model, device, geometry, "clean") for f, model in enumerate(models)]
        member = [reconstruct_identity(r, size, geometry) for r in predictions]
        legacy = p.average_five([reconstruct(r, size, geometry).astype(np.float32) for r in predictions])
    else:
        cfg = replace(geometry, coarse_side=geometry.patch_side)
        predictions = [predict_regions(image, model, None, device, cfg, "clean") for model in models]
        member = [reconstruct(r, size, cfg).astype(np.float32) for r in predictions]; legacy = None
    return p.average_five(member), member, predictions, legacy


def validate_parents(policy, previous, controls, prepared, archive, store):
    for label, root, status in (("external", previous, "source_external_complete"), ("controls", controls, "source_external_controls_complete")):
        root = Path(root); done = read(root / "completed.json"); expected = policy["parents"][label]
        if sha256(root / "completed.json") != expected["completion_sha256"] or done["status"] != status or done["protocol_sha256"] != expected["protocol_sha256"]:
            raise ValueError("Reviewed parent differs:" + label)
        p.verify_files(root, done["file_sha256"])
        if sha256(root / "analysis_protocol.json") != expected["protocol_sha256"]: raise ValueError("Parent protocol differs")
        old = read(root / "analysis_protocol.json")
        if old["cases"] != policy["cases"] or old["geometry"] != policy["geometry"]: raise ValueError("Cohort/geometry differs")
        print(f"Verified {label} parent receipts/arrays;no model rerun", flush=True)
    checked = p.verify_inputs(read(Path(previous) / "analysis_protocol.json"), prepared, archive, store)
    prior_controls = read(Path(controls) / "analysis_protocol.json")
    if prior_controls["members"] != policy["members"]: raise ValueError("Frozen weights differ from reviewed controls")
    guard = read(Path(controls) / "diagnostics/completed.json")
    if not guard["guard_passed"]: raise ValueError("Previous technical diagnostic not passed")
    # The prior full file receipt already covers its diagnostic record. Re-check
    # every actually used weight, including boundary absent from parent25.
    for method in set(WEIGHTS.values()):
        for f, member in enumerate(policy["members"][method]):
            run = run_path(store, method, f, 17); expected = member["receipt"]["primary"]
            p.verify_files(run, {"identity.json": member["identity_file_sha256"], "completed.json": member["completion_file_sha256"], "best.pt": expected["checkpoint_sha256"]})
            ident, done = read(run / "identity.json"), read(run / "completed.json")
            if ident != expected["identity"] or done["identity"] != ident or done["best_sha256"] != expected["checkpoint_sha256"] or member["fold"] != f or member["seed"] != 17:
                raise ValueError("Wrong member identity")
    checked.update(private_patient_pixels_read=False, private54_pixels_read=False, technical_positive_checks_reused=True,
                   current_used_checkpoints=25, previous_guard_passed=True)
    return checked


def probability_guard(legacy, new, predictions, old_path, direct_path, shape, tolerance):
    saved = p.probability(old_path, shape); replay = c.match_probabilities(legacy, saved, tolerance)
    all_full = all(single_fullframe(r) for r in predictions)
    identity = c.match_probabilities(new, p.probability(direct_path, shape), tolerance) if all_full else None
    if not replay["passed"] or (identity is not None and not identity["passed"]):
        raise ValueError("Frozen cascade replay/fullframe identity mismatch;no tolerance adjustment")
    return dict(legacy_replay=replay, all_five_single_fullframe=all_full, fullframe_identity_check=identity,
                single_fullframe_folds=sum(single_fullframe(r) for r in predictions))


def store_case(dest, mean, member, predictions, gt, case, method, digest, elapsed, guard):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "probability.npy", mean.astype(np.float32), allow_pickle=False)
    Image.fromarray((mean >= .5).astype(np.uint8) * 255).save(dest / "prediction.png")
    done = dict(protocol_sha256=digest, method=method, patient_id=case["patient_id"], metrics=p.metrics_row(mean, gt, case, method),
                member_metrics=[p.metrics_row(q, gt, case, method, f) for f, q in enumerate(member)],
                file_sha256={n: sha256(dest / n) for n in ("probability.npy", "prediction.png")}, pipeline_predictions=5,
                network_forward_calls=sum(len(r["patch_probabilities"]) for r in predictions) if method == IDENTITY else 5,
                proposals=[dict(fold=f, boxes=r["boxes"], raw_component_count=r.get("raw_component_count"), proposal_mode=r.get("proposal_mode")) for f, r in enumerate(predictions)] if method == IDENTITY else [],
                reconstruction_guard=guard, original_input_size=list(predictions[0]["native_size"]), reference_read_after_RGB_inference=True,
                cached_inference_seconds=elapsed, timing_comparable_across_methods=False,
                actual_weight_method=WEIGHTS[method], direct_input_side=512 if method != IDENTITY else None)
    save(dest / "completed.json", done); return done


def summaries(rows, policy):
    ids = sorted(q["patient_id"] for q in policy["cases"]); lookup = {}
    for r in rows:
        key = (r["method"], r["patient_id"])
        if key in lookup: raise ValueError("Duplicate patient-method")
        lookup[key] = r
    if set(lookup) != {(method, pid) for method in METHODS for pid in ids}: raise ValueError("Incomplete paired cohort")
    groups = [("all", ids)]
    for field in ("country", "source_encoding"):
        for v in sorted({q[field] for q in policy["cases"]}): groups.append((field + ":" + v, sorted(q["patient_id"] for q in policy["cases"] if q[field] == v)))
    summary = []; events = []
    for group, patients in groups:
        for method in METHODS:
            records = [lookup[(method, pid)] for pid in patients]
            for metric in p.METRIC_NAMES:
                v = np.array([r[metric] for r in records], np.float64)
                row = dict(group=group, method=method, metric=metric, n_patients=len(v), mean=float(v.mean()), median=float(np.median(v)), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.)
                if group == "all" and metric == "dice":
                    _, lo, hi = c.bootstrap_mean_ci(v, policy["bootstrap"], policy["statistics_seed"]); row.update(ci95_lower=lo, ci95_upper=hi)
                summary.append(row)
    vectors = {method: np.array([lookup[(method, pid)]["dice"] for pid in ids]) for method in METHODS}; contrasts = []
    pairs = [(DIRECT, method) for method in NEW_METHODS if method != IDENTITY]
    pairs += [(IDENTITY, "local_combined"), (DIRECT, IDENTITY), ("fullframe_globalunet_512", "global_unet_768")]
    for candidate, control in pairs:
        a, b = vectors[candidate], vectors[control]
        contrasts.append(dict(contrast="paired_Dice", candidate=candidate, control=control,
                              **c.paired_bootstrap(a, b, policy["bootstrap"], policy["statistics_seed"]),
                              p_two_sided_signflip=c.sign_flip_pvalue(a - b, policy["permutations"], policy["statistics_seed"])))
    delta = vectors[DIRECT] - vectors["fullframe_boundary_512"] - vectors["fullframe_consistency_512"] + vectors["fullframe_plain_512"]
    avg, lo, hi = c.bootstrap_mean_ci(delta, policy["bootstrap"], policy["statistics_seed"])
    contrasts.append(dict(contrast="fullframe_factorial_interaction", candidate="C-B-K+F_at512", control="zero", mean_delta_dice=avg, ci95_lower=lo, ci95_upper=hi,
                          n_patients=len(ids), p_two_sided_signflip=c.sign_flip_pvalue(delta, policy["permutations"], policy["statistics_seed"])))
    for row, adjusted in zip(contrasts, c.holm_adjust([r["p_two_sided_signflip"] for r in contrasts])):
        row.update(holm_p_across8=float(adjusted), interpretation="post_result_exploratory;conditional_fitted_weights;not independent validation or training-randomness inference")
    for method in METHODS:
        records = [lookup[(method, pid)] for pid in ids]
        for event, number in (("empty_prediction", sum(int(r["empty_prediction"]) for r in records)), ("dice_below_0_5", sum(r["dice"] < .5 for r in records))):
            rate, lo, hi = c.wilson_rate_ci(number, len(ids)); events.append(dict(method=method, event=event, events=number, n_patients=len(ids), rate=rate, wilson_ci95_lower=lo, wilson_ci95_upper=hi))
    return summary, contrasts, events


def collect_return(output):
    output = Path(output)
    with zipfile.ZipFile(output / "public_mechanism_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(output.rglob("*")):
            if f.is_file() and (f.suffix in (".csv", ".json", ".md") or f.parent.name == "qc"): z.write(f, f.relative_to(output).as_posix())


def finish(output, digest, status, details):
    output = Path(output); save(output / "summary.json", dict(status=status, **details))
    files = [f for f in output.rglob("*") if f.is_file() and f.suffix != ".zip" and f != output / "completed.json"]
    returned = [f for f in files if f.suffix in (".csv", ".json", ".md") or f.parent.name == "qc"]
    done = dict(status=status, protocol_sha256=digest, completed_utc=datetime.now(timezone.utc).isoformat(), training_fits=0,
                file_sha256={f.relative_to(output).as_posix(): sha256(f) for f in sorted(files)},
                returned_file_sha256={f.relative_to(output).as_posix(): sha256(f) for f in sorted(returned)}, arrays_kept_server_only=True, private_patient_pixels_read=False)
    save(output / "completed.json", done); collect_return(output); return done


def qc_sheets(output, prepared, controls, cases):
    output = Path(output); (output / "qc").mkdir(exist_ok=True); manifest = []
    show = ((DIRECT, "Fine C512"), ("fullframe_plain_512", "Fine plain512"), ("fullframe_boundary_512", "Fine boundary512"),
            ("fullframe_consistency_512", "Fine consistency512"), ("fullframe_globalunet_512", "Global U @512"), (IDENTITY, "Cascade identity"))
    for start in range(0, len(cases), 12):
        canvas = Image.new("RGB", (1820, 1176), "white"); subset = cases[start:start + 12]
        for i, case in enumerate(subset):
            with Image.open(p.contained(prepared, case["image_path"])) as im: rgb = np.array(im.convert("RGB"))
            gt = p.reference_mask(prepared, case); yy, xx = np.nonzero(gt)
            box = (max(0, int(xx.min()) - 24), max(0, int(yy.min()) - 24), min(gt.shape[1], int(xx.max()) + 25), min(gt.shape[0], int(yy.max()) + 25))
            tile = Image.new("RGB", (910, 196), "white"); draw = ImageDraw.Draw(tile); draw.text((2, 0), case["patient_id"] + " | crop DISPLAY ONLY", fill="black")
            ref = rgb.copy(); ref[gt] = (.65 * rgb[gt] + .35 * np.array([0, 220, 80])).astype(np.uint8); overlays = [("Reference", ref)]
            for method, label in show:
                root = controls if method == DIRECT else output
                pred = p.probability(Path(root) / "methods" / method / case["patient_id"] / "probability.npy", gt.shape) >= .5
                a = rgb.copy(); a[pred] = (.65 * rgb[pred] + .35 * np.array([0, 100, 255])).astype(np.uint8); a[gt & ~ndi.binary_erosion(gt)] = (255, 220, 0); overlays.append((label, a))
            for j, (label, a) in enumerate(overlays):
                thumb = Image.fromarray(a).crop(box); thumb.thumbnail((128, 160), Image.Resampling.LANCZOS)
                draw.text((j * 130 + 1, 17), label, fill="black"); tile.paste(thumb, (j * 130, 32))
            canvas.paste(tile, ((i % 2) * 910, (i // 2) * 196))
        name = f"qc/sheet_{start // 12 + 1:02d}.png"; canvas.save(output / name)
        manifest.append(dict(file=name, patient_ids=[r["patient_id"] for r in subset], GT_crop_display_only=True))
    save(output / "qc_manifest.json", manifest)


def evaluate(policy, digest, prepared, archive, store, previous, controls, output, device, loader=load_network):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        done = read(output / "completed.json")
        if done["protocol_sha256"] != digest: raise ValueError("Completed protocol differs")
        p.verify_files(output, done["file_sha256"]); collect_return(output); return done
    geometry = CascadeConfig(**policy["geometry"]); records = []
    for method in c.METHODS:
        source = Path(previous) if method in p.METHODS else Path(controls)
        label = "external" if method in p.METHODS else "controls"
        for case in policy["cases"]:
            record = p.case_done(source / "methods" / method / case["patient_id"], policy["parents"][label]["protocol_sha256"], method, case)
            if record is None: raise ValueError("Missing reviewed parent case")
            records.append(record)
    with zipfile.ZipFile(archive) as z:
        for method in NEW_METHODS:
            pending = [q for q in policy["cases"] if not p.case_done(output / "methods" / method / q["patient_id"], digest, method, q)]
            key = WEIGHTS[method]; family = REFINER_FAMILIES[key] if key.startswith("local_") else global_spec(key)[1]
            models = [loader(run_path(store, key, f, 17) / "best.pt", policy["members"][key][f]["receipt"]["primary"], family, device) for f in range(5)] if pending else []
            for i, case in enumerate(policy["cases"], 1):
                pid = case["patient_id"]; dest = output / "methods" / method / pid; record = p.case_done(dest, digest, method, case)
                if record: records.append(record); continue
                start = time.perf_counter(); image = p.original_rgb(z, case)
                coarse = p.coarse_predictions(Path(previous) / "methods/global_unet_768" / pid, image) if method == IDENTITY else None
                mean, member, predictions, legacy = infer(image, method, models, coarse, device, case["reference_size"], geometry); guard = None
                if method == IDENTITY:
                    try:
                        guard = probability_guard(legacy, mean, predictions, Path(previous) / "methods/local_combined" / pid / "probability.npy",
                                                  Path(controls) / "methods" / DIRECT / pid / "probability.npy", tuple(reversed(case["reference_size"])), policy["replay_tolerance"])
                    except ValueError as error:
                        save(output / "guard_failure.json", dict(patient_id=pid, method=method, message=str(error)))
                        return finish(output, digest, "reconstruction_guard_failed", dict(training_fits=0, complete_comparisons=False, stopped_patient_id=pid))
                gt = p.reference_mask(prepared, case)
                record = store_case(dest, mean, member, predictions, gt, case, method, digest, time.perf_counter() - start, guard); records.append(record)
                if i % 12 == 0 or i == len(policy["cases"]): print(f"Fixed mechanism {method}:{i}/{len(policy['cases'])};0fits", flush=True)
            del models; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
            collect_return(output)
    rows = [d["metrics"] for d in records]; members = [r for d in records for r in d["member_metrics"]]
    summary, contrasts, events = summaries(rows, policy)
    for name, values in (("patient_metrics.csv", rows), ("member_metrics.csv", members), ("summary_metrics.csv", summary), ("paired_dice_contrasts.csv", contrasts), ("failure_events.csv", events)):
        csv_write(output / name, values)
    qc_sheets(output, prepared, controls, policy["cases"])
    (output / "README.md").write_text("Post-result exploratory source-external ablations;0new fits;fixed212sourceunion cases;original RGB inputs,GT for scoring only,threshold0.5. Direct512 fine plain/boundary/consistency compared to reused direct combined512. Whole-image-trained U768 weights evaluated at512 solely as an inference-resolution intervention,not retrained512 baseline. Single exact fullframe ROI reconstruction preserves fine probability;all other proposals/reconstruction unchanged. The original combined cascade is replayed and checked on every case;all5fullframe cases must equal direct fine ensemble. NewHolm8 conditional-patient comparisons;parentHolm5/Holm10 unchanged. No private patient pixels. This packet closes the predeclared inference mechanism comparison;no automatic training,threshold/scale/weight searches or source swapping.\n", encoding="utf8")
    return finish(output, digest, "public_mechanism_complete", dict(training_fits=0, n_patients=len(policy["cases"]), methods=list(METHODS), new_methods=list(NEW_METHODS),
                  patient_method_rows=len(rows), member_rows=len(members), new_network_forward_calls=sum(d["network_forward_calls"] for d in records if d["method"] in NEW_METHODS),
                  new_pipeline_predictions=sum(d["pipeline_predictions"] for d in records if d["method"] in NEW_METHODS),
                  private_patient_pixels_read=False, original54_recomputed=False, public_fitting=False, threshold_search=False, GT_crop_inference=False,
                  all_source_patients_retained=True, source_reference_changed=False, comparisons_post_result_exploratory=True))


def main():
    ap = argparse.ArgumentParser(description="Frozen zero-fit source-external mechanism ablations")
    ap.add_argument("--archive", default="public_datasets/eyes_defy_anemia.zip"); ap.add_argument("--prepared-public", default="results_public_reference_prepare_v1")
    ap.add_argument("--model-store", default="results_round4"); ap.add_argument("--previous", default="results_public_external_v1"); ap.add_argument("--controls", default="results_public_external_controls_v1")
    ap.add_argument("--output", default="results_public_mechanism_v1"); ap.add_argument("--protocol", required=True); ap.add_argument("--protocol-sha256", required=True); ap.add_argument("--cpu-threads", type=int, default=4)
    args = ap.parse_args(); spec = Path(args.protocol).resolve(); digest = sha256(spec)
    if digest != args.protocol_sha256: raise ValueError("Frozen mechanism protocol changed")
    policy = read(spec); p.verify_files(Path(__file__).resolve().parents[2], policy["source_sha256"])
    if policy["schema"] != "conjseg_public_mechanism_v1" or policy["methods"] != list(METHODS) or policy["new_methods"] != list(NEW_METHODS) or policy["training_fits"] != 0 or len(policy["cases"]) != 212 or policy["threshold"] != .5:
        raise ValueError("Wrong fixed plan")
    output = Path(args.output).resolve()
    for source in (args.archive, args.prepared_public, args.model_store, args.previous, args.controls, spec):
        source = Path(source).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output): raise ValueError("Output overlaps protected source")
    if output.exists() and any(output.iterdir()) and not (output / "analysis_protocol.json").exists(): raise ValueError("Unrelated output directory")
    if args.cpu_threads < 1: raise ValueError("Positive CPU thread count required")
    verification = validate_parents(policy, args.previous, args.controls, args.prepared_public, args.archive, args.model_store)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA not visible in selected existing Python;no installation performed")
    output.mkdir(parents=True, exist_ok=True); frozen = output / "analysis_protocol.json"
    if frozen.exists() and frozen.read_bytes() != spec.read_bytes(): raise ValueError("Partial protocol changed")
    frozen.write_bytes(spec.read_bytes())
    if not (output / "completed.json").exists():
        save(output / "input_verification.json", verification); torch.set_num_threads(args.cpu_threads); seed_all(17)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        environment = actual_environment(); path = output / "environment.json"
        if path.exists() and read(path) != environment: raise ValueError("Partial execution environment changed")
        save(path, environment)
    done = evaluate(policy, digest, args.prepared_public, args.archive, args.model_store, args.previous, args.controls, output, torch.device("cuda"))
    print(f"{done['status']};return {output / 'public_mechanism_results.zip'};0fits", flush=True)


if __name__ == "__main__": main()
