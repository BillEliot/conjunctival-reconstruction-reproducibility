"""Six bounded local-refinement fits. Development only; no environment changes."""
import argparse
import gc
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from . import __version__
from .audit import save_json, sha256, verify_prepared
from .cascade import CascadeConfig, context_box, predict_regions, reconstruct, propose_regions
from .data import labelme_mask, letterbox, read_json, load_cached
from .evaluation import STRESSES, crop
from .metrics import compute_metrics
from .models import Segmenter
from .round2 import completed_identity, overlap, review_plate, source_case, validation_rows
from .stats import paired_bootstrap
from .training import seed_all, worker_seed, photometric, seg_loss, edge_loss, masked_mean

SEEDS = (17, 29, 43)
METHODS = ("coarse_unet", "local_unet", "local_combined")
REFINER_FAMILIES = {"local_unet": "unet", "local_boundary": "boundary", "local_consistency": "consistency", "local_combined": "proposed"}


@dataclass(frozen=True)
class RefinerConfig:
    epochs: int = 60
    patience: int = 12
    batch_size: int = 6
    workers: int = 4
    lr: float = .0003
    weight_decay: float = .0001
    boundary_weight: float = .2
    consistency_weight: float = .5
    consistency_ramp_epochs: int = 15
    positive_crop_probability: float = .8
    full_frame_probability: float = .1
    train_context_range: tuple = (2., 3.)
    train_center_jitter: float = .2


def open_rgb(images, row):
    # Sources are hash-checked before training; never rotate annotation coordinates.
    with Image.open(Path(images)/str(row["image_relpath"]).replace("\\", "/")) as original:
        if original.getexif().get(274, 1) != 1 or original.size != (int(row["width"]), int(row["height"])):
            raise ValueError("Image geometry differs from frozen manifest")
        return original.convert("RGB")


class TrainingPatches(Dataset):
    """One item per TRAINING patient per epoch, including bilateral photographs.

    Reference-guided patch sampling is exclusively a training operation. Held-out
    validation-fold and final holdout patients are rejected even if supplied accidentally.
    """
    def __init__(self, images, frame, config=RefinerConfig(), geometry=CascadeConfig(), validation_fold=0, memory=None):
        if validation_fold not in range(5) or frame.empty or frame.patient_id.duplicated().any() or not frame.split.eq("development").all() or not frame.fold.isin(range(5)).all() or frame.fold.eq(validation_fold).any():
            raise ValueError("Patch dataset requires unique development TRAINING patients outside the validation fold")
        self.images, self.rows, self.config, self.geometry = Path(images), frame.to_dict("records"), config, geometry
        self.memory = memory

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        if self.memory is None:
            image = open_rgb(self.images, row)
            mask = labelme_mask(read_json(self.images / str(row["annotation_relpath"]).replace("\\", "/")), image.size)
            size = image.size
        else:
            a = self.memory.get(row); size = a["size"]
        u = random.random()
        if u < self.config.positive_crop_probability:
            if self.memory is None:
                labels, _ = ndi.label(np.asarray(mask) > 0, structure=np.ones((3, 3)))
                sy, sx = random.choice(ndi.find_objects(labels))
                component = (sx.start, sy.start, sx.stop, sy.stop)
            else:
                component = random.choice(a["components"])
            bounds = context_box(component, size,
                factor=random.uniform(*self.config.train_context_range), minimum=self.geometry.minimum_native_side,
                jitter=(random.uniform(-self.config.train_center_jitter, self.config.train_center_jitter), random.uniform(-self.config.train_center_jitter, self.config.train_center_jitter)))
        elif u < self.config.positive_crop_probability + self.config.full_frame_probability:
            bounds = (0, 0, *size)
        else:
            # Random location; it may contain tissue. ALWAYS keep its true mask.
            side = min(min(size), max(self.geometry.minimum_native_side, round(random.uniform(.15, .4)*min(size))))
            l, t = random.randint(0, size[0]-side), random.randint(0, size[1]-side)
            bounds = (l, t, l+side, t+side)
        patch, patch_mask = ((image.crop(bounds), mask.crop(bounds)) if self.memory is None else self.memory.patch(row, bounds))
        rgb, m, valid, _ = letterbox(patch, patch_mask, self.geometry.patch_side)
        x = rgb.astype(np.float32)/255
        if random.random() < .5:
            x, m, valid = x[:, ::-1].copy(), m[:, ::-1].copy(), valid[:, ::-1].copy()
        edge = (ndi.binary_dilation(m > 0, iterations=2) ^ ndi.binary_erosion(m > 0, iterations=2)).astype(np.float32)*valid
        return dict(image=torch.from_numpy(x.transpose(2, 0, 1).copy()), mask=torch.from_numpy(m[None].copy()),
                    valid=torch.from_numpy(valid[None].copy()), edge=torch.from_numpy(edge[None]), patient_id=row["patient_id"])


def coarse_receipt(run, prepared, frame, seed):
    identity, done = completed_identity(run, prepared, frame)
    if identity["family"] != "unet" or identity["seed"] != seed or identity["config"]["side"] != 768:
        raise ValueError("Expected matching-seed 768 development U-Net coarse checkpoint")
    return {"identity": identity, "checkpoint_sha256": done["best_sha256"]}


def load_coarse(run, receipt, device):
    ck = torch.load(Path(run)/"best.pt", map_location="cpu", weights_only=False)
    if ck["identity"] != receipt["identity"]:
        raise ValueError("Coarse checkpoint embedded identity changed")
    model = Segmenter("unet", pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval().requires_grad_(False)
    return model, ck["model"]


@torch.inference_mode()
def validation_score(images, prepared1024, rows, coarse, refiner, device, geometry):
    refiner.eval()
    scores = []
    for row in rows.to_dict("records"):
        image = open_rgb(images, row)
        cached = load_cached(prepared1024, row)
        gt = crop(cached["mask"] > 0, cached["box"])
        prediction = predict_regions(image, coarse, refiner, device, geometry)
        p = reconstruct(prediction, (gt.shape[1], gt.shape[0]), geometry)
        scores.append(overlap(p >= geometry.output_threshold, gt)["dice"])
    return float(np.mean(scores))


def train_refiner(images, prepared, prepared1024, frame, output, method, seed, coarse_run, config, geometry, resume, fold=0, stage="round3_development", supplied_receipt=None, runtime=None):
    if method not in REFINER_FAMILIES:
        raise ValueError("Unexpected refiner method")
    device = torch.device("cuda")
    if stage not in ("round3_development", "formal_development") or fold not in range(5):
        raise ValueError("Only development folds may train local refinement")
    receipt = coarse_receipt(coarse_run, prepared, frame, seed) if supplied_receipt is None else supplied_receipt
    ci = receipt["identity"]
    expected_stage = "pilot" if stage == "round3_development" else "formal_development"
    if ci["stage"] != expected_stage or ci["fold"] != fold or ci["seed"] != seed or ci["family"] != "unet" or ci["config"]["side"] != geometry.coarse_side or ci["manifest_sha256"] != sha256(Path(prepared)/"manifest.csv"):
        raise ValueError("Coarse receipt differs in stage, fold, seed, family or data")
    if sha256(Path(coarse_run)/"best.pt") != receipt["checkpoint_sha256"]:
        raise ValueError("Coarse weight hash changed")
    tr = frame[(frame.split == "development") & (frame.fold != fold)]
    va = validation_rows(frame, fold)
    out = (Path(output)/"runs"/method/f"seed_{seed}" if stage == "round3_development" else
           Path(output)/"runs"/stage/method/f"fold_{fold}"/f"seed_{seed}")
    out.mkdir(parents=True, exist_ok=True)
    # Convert tuple-valued config to JSON-native lists before storing/comparing.
    from .performance import FORMAL_PROTOCOL_VERSION, CheckpointWriter, timed_batches, write_profile
    identity = json.loads(json.dumps(dict(code_version=FORMAL_PROTOCOL_VERSION if stage == "formal_development" else __version__, stage=stage, fold=fold, seed=seed,
        method=method, config=asdict(config), geometry=asdict(geometry),
        manifest_sha256=sha256(Path(prepared)/"manifest.csv"), reference_manifest_sha256=sha256(Path(prepared1024)/"manifest.csv"),
        coarse=receipt, train_patient_ids=sorted(tr.patient_id), validation_patient_ids=sorted(va.patient_id))))
    best_path, last_path = out/"best.pt", out/"last.pt"
    if (out/"completed.json").exists():
        done = read_json(out/"completed.json")
        if done["identity"] != identity or sha256(best_path) != done["best_sha256"]:
            raise ValueError("Completed refiner identity/weight hash changed")
        return out
    if last_path.exists() and not resume:
        raise ValueError("Unfinished refiner exists; use --resume")
    seed_all(seed)
    coarse, initial = load_coarse(coarse_run, receipt, device)
    family = REFINER_FAMILIES[method]
    model = Segmenter(family, pretrained=False).to(device)
    missing, unexpected = model.load_state_dict(initial, strict=False)
    if unexpected or set(missing) != ({"edge_head.0.weight", "edge_head.0.bias", "edge_head.2.weight", "edge_head.2.bias"} if model.use_boundary else set()):
        raise ValueError("Only the new auxiliary edge head may be missing from coarse initialization")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda")
    start, best, stale, history = 0, -1., 0, []
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        if ck["identity"] != identity:
            raise ValueError("Resume identity/config/dataset changed")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best, stale, history = ck["epoch"]+1, ck["best"], ck["stale"], ck["history"]
        random.setstate(ck["python_rng"]); np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state_all(ck["cuda_rng"])
    save_json(out/"identity.json", identity)
    dataset = TrainingPatches(images, tr, config, geometry, validation_fold=fold, memory=runtime.native if runtime is not None else None)
    validator = runtime.validator(prepared1024, va, coarse, device, geometry, receipt) if runtime is not None else None
    if validator is not None:
        validator.verify_refiner(images, va, coarse, model, device, out)
    writer = CheckpointWriter(out) if runtime is not None else None; next_epoch = None
    try:
        for epoch in range(start, config.epochs):
            if stale >= config.patience:
                break
            generator = torch.Generator().manual_seed(seed*10000+epoch)
            if next_epoch is None:
                loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers,
                    pin_memory=True, worker_init_fn=worker_seed, generator=generator, **(runtime.loader_options(config.workers) if runtime is not None else {}))
                batches = iter(loader)
            else:
                loader, batches = next_epoch; next_epoch = None
            model.train(); losses = []; t0 = time.perf_counter(); profile = dict(data_wait_seconds=0.)
            for batch in tqdm(timed_batches(batches, profile), total=len(loader), desc=f"{stage}/{method}/f{fold}/s{seed}/e{epoch+1}"):
                x, m, v, edge = [batch[k].to(device, non_blocking=True) for k in ("image", "mask", "valid", "edge")]
                x2 = photometric(x, v)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    l1, e1 = model(x); l2, e2 = model(x2)
                    loss = .5*(seg_loss(l1, m, v)+seg_loss(l2, m, v))
                    if model.use_boundary:
                        loss += config.boundary_weight*.5*(edge_loss(e1, edge, v)+edge_loss(e2, edge, v))
                    if model.use_consistency:
                        ramp = min(1., (epoch+1)/config.consistency_ramp_epochs)
                        loss += config.consistency_weight*ramp*masked_mean((l2.float().sigmoid()-l1.float().sigmoid().detach()).square(), v*(1+4*m+2*edge))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite local-refinement training loss")
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update()
                losses.append(float(loss.detach()))
            train_seconds = time.perf_counter()-t0; val_start = time.perf_counter()
            score = (validation_score(images, prepared1024, va, coarse, model, device, geometry) if validator is None else validator.score(model))
            validation_seconds = time.perf_counter()-val_start
            if not np.isfinite(score):
                raise FloatingPointError("Non-finite full-pipeline validation score")
            scheduler.step()
            better = score > best
            if better:
                best, stale = score, 0
                if writer is None:
                    torch.save(dict(model=model.state_dict(), identity=identity, epoch=epoch, val_dice=score), best_path)
            else:
                stale += 1
            row = dict(epoch=epoch+1, train_loss=float(np.mean(losses)), val_dice=score, best_dice=best,
                       seconds=time.perf_counter()-t0, lr=optimizer.param_groups[0]["lr"])
            history.append(row)
            ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                identity=identity, epoch=epoch, best=best, stale=stale, history=history, python_rng=random.getstate(),
                numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            if writer is None:
                tmp = out/"last.tmp.pt"; torch.save(ck, tmp); os.replace(tmp, last_path)
            else:
                writer.wait()
                if stale < config.patience and epoch+1 < config.epochs:
                    generator_next = torch.Generator().manual_seed(seed*10000+epoch+1)
                    loader_next = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True,
                        worker_init_fn=worker_seed, generator=generator_next, **runtime.loader_options(config.workers))
                    next_epoch = (loader_next, iter(loader_next))
                writer.submit(ck, better)
                write_profile(out, epoch, profile, train_seconds, validation_seconds)
            pd.DataFrame(history).to_csv(out/"history.csv", index=False)
            print(json.dumps(row), flush=True)
    finally:
        next_epoch = None
        if writer is not None:
            writer.close()
    # Restore history from checkpoint on resume, including crashes between receipt writes.
    pd.DataFrame(history).to_csv(out/"history.csv", index=False)
    best_ck = torch.load(best_path, map_location="cpu", weights_only=False)
    save_json(out/"completed.json", dict(identity=identity, best_sha256=sha256(best_path),
        best_epoch=best_ck["epoch"]+1, best_val_dice=best_ck["val_dice"], holdout_used=False))
    del coarse, model, optimizer, initial, ck, best_ck
    gc.collect(); torch.cuda.empty_cache()
    return out


def proposal_coverage(boxes, gt):
    """Reference comparison AFTER proposal generation; never used by inference."""
    coverage = np.zeros(gt.shape, bool)
    for l, t, r, b in boxes:
        coverage[t:b, l:r] = True
    labels, n = ndi.label(gt, structure=np.ones((3, 3)))
    component_coverage = [float(coverage[labels == i].mean()) for i in range(1, n+1)]
    return dict(proposal_gt_pixel_coverage=float(coverage[gt].mean()),
        proposal_gt_component_coverage_min=min(component_coverage), proposal_area_fraction=float(coverage.mean()))


@torch.inference_mode()
def diagnose(images, prepared, prepared1024, frame, output, method, seed, coarse_run, refiner_run, geometry):
    device = torch.device("cuda")
    receipt = coarse_receipt(coarse_run, prepared, frame, seed)
    ref_done = read_json(Path(refiner_run)/"completed.json") if refiner_run is not None else None
    commitment = dict(method=method, seed=seed, coarse=receipt, refiner=ref_done, geometry=asdict(geometry),
        manifest_sha256=sha256(Path(prepared)/"manifest.csv"), reference_manifest_sha256=sha256(Path(prepared1024)/"manifest.csv"),
        transforms=["clean", *STRESSES], holdout_used=False, timing="decoded RGB to common-grid probability, including resize, proposals, fusion; excluding IO, GT metrics and native reconstruction; float32 batch1")
    dest = Path(output)/"diagnostics"/method/f"seed_{seed}"
    if (dest/"completed.json").exists():
        done = read_json(dest/"completed.json")
        if done["commitment"] != commitment or sha256(dest/"patient_metrics.csv") != done["metrics_sha256"]:
            raise ValueError("Diagnostic commitment/metrics changed")
        for plate in done["review_plates"]:
            if sha256(dest/plate["path"]) != plate["sha256"]:
                raise ValueError("Review plate changed")
        return pd.read_csv(dest/"patient_metrics.csv")
    dest.mkdir(parents=True, exist_ok=True); (dest/"review_plates").mkdir(exist_ok=True)
    coarse, _ = load_coarse(coarse_run, receipt, device)
    refiner = None
    if ref_done is not None:
        if ref_done["identity"]["coarse"] != receipt or sha256(Path(refiner_run)/"best.pt") != ref_done["best_sha256"]:
            raise ValueError("Refiner was trained against a different coarse checkpoint")
        ck = torch.load(Path(refiner_run)/"best.pt", map_location="cpu", weights_only=False)
        if ck["identity"] != ref_done["identity"]:
            raise ValueError("Refiner checkpoint embedded identity changed")
        refiner = Segmenter("unet" if method == "local_unet" else "proposed", pretrained=False).to(device).eval()
        refiner.load_state_dict(ck["model"])
    rows = []
    val = validation_rows(frame)
    for i, row in enumerate(val.to_dict("records")):
        image = open_rgb(images, row)
        cached = load_cached(prepared1024, row)
        gt = crop(cached["mask"] > 0, cached["box"])
        if i == 0:
            for _ in range(3):
                warm = predict_regions(image, coarse, refiner, device, geometry)
                reconstruct(warm, (gt.shape[1], gt.shape[0]), geometry)
        for transform in ("clean", *STRESSES):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            prediction = predict_regions(image, coarse, refiner, device, geometry, transform)
            probability = reconstruct(prediction, (gt.shape[1], gt.shape[0]), geometry)
            torch.cuda.synchronize(); elapsed = (time.perf_counter()-t0)*1000
            # Baseline timing contains only its actual inference path. Proposal
            # diagnostics for the baseline are computed after the timer stops.
            if refiner is None:
                prediction["boxes"], proposal_meta = propose_regions(prediction["coarse_content"], image.size, geometry)
                prediction.update(proposal_meta)
            met = compute_metrics(probability >= geometry.output_threshold, gt) if transform == "clean" else overlap(probability >= geometry.output_threshold, gt)
            met.update(patient_id=row["patient_id"], method=method, seed=seed, transform=transform,
                n_polygons=row["n_polygons"], has_dng=row["has_dng"], reference_side=geometry.reference_side,
                n_regions=len(prediction["boxes"]), raw_component_count=prediction["raw_component_count"],
                n_refined_regions=len(prediction["patch_probabilities"]),
                proposal_mode=prediction["proposal_mode"], pipeline_wall_ms=elapsed, native_dice=np.nan,
                total_parameters=sum(p.numel() for p in coarse.parameters()) + (sum(p.numel() for p in refiner.parameters()) if refiner is not None else 0))
            if transform == "clean":
                _, native_mask = source_case(images, row)
                native_gt = np.asarray(native_mask) > 0
                native_pred = reconstruct(prediction, image.size, geometry) >= geometry.output_threshold
                met["native_dice"] = overlap(native_pred, native_gt)["dice"]
                met.update(proposal_coverage(prediction["boxes"], native_gt))
                met["proposal_boxes_json"] = json.dumps(prediction["boxes"])
            rows.append(met)
        print(f"Round3 diagnostics {method}/s{seed}: {i+1}/{len(val)}", flush=True)
    data = pd.DataFrame(rows)
    if len(data) != len(val)*7 or data.duplicated(["patient_id", "transform"]).any():
        raise ValueError("Diagnostic row count/identity mismatch")
    ranking = data[data["transform"] == "clean"].sort_values(["dice", "patient_id"])
    selected = {pid: "worst3" for pid in ranking.head(3).patient_id}
    selected.setdefault(ranking.iloc[len(ranking)//2].patient_id, "median")
    selected.setdefault(ranking.iloc[-1].patient_id, "best")
    two = ranking[ranking.n_polygons == 2]
    if len(two):
        selected.setdefault(two.iloc[0].patient_id, "worst_two_polygon")
    plates = []
    for row in val[val.patient_id.isin(selected)].to_dict("records"):
        image, mask = source_case(images, row)
        prediction = predict_regions(image, coarse, refiner, device, geometry)
        pred = reconstruct(prediction, image.size, geometry) >= geometry.output_threshold
        path = dest/"review_plates"/f"{row['patient_id']}.jpg"
        review_plate(image, np.asarray(mask) > 0, pred, row["patient_id"], path)
        plates.append(dict(path=str(path.relative_to(dest)), sha256=sha256(path), patient_id=row["patient_id"], selection=selected[row["patient_id"]]))
    data.to_csv(dest/"patient_metrics.csv", index=False)
    save_json(dest/"completed.json", dict(commitment=commitment, metrics_sha256=sha256(dest/"patient_metrics.csv"), review_plates=plates,
        statistical_unit="patient; timing is one measurement per patient-condition; seed/view repeats not independent",
        coverage="Post-inference reference comparison; no GT proposal inputs", annotation_crops="display only after full inference"))
    del coarse, refiner
    gc.collect(); torch.cuda.empty_cache()
    return data


def summarize(frames, destination):
    dest = Path(destination); dest.mkdir(exist_ok=True)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["method", "seed", "patient_id", "transform"]).any():
        raise ValueError("Duplicate result rows")
    patient_sets = data.groupby(["method", "seed", "transform"]).patient_id.apply(lambda s: tuple(sorted(s)))
    if len(set(patient_sets)) != 1:
        raise ValueError("Methods/seeds/conditions have different patient IDs")
    avg = data.groupby(["method", "patient_id", "transform"]).agg(dice=("dice", "mean"), native_dice=("native_dice", "mean"),
        n_seeds=("seed", "nunique"), pipeline_wall_ms=("pipeline_wall_ms", "mean"), n_regions=("n_regions", "mean"),
        surface_dice_0_005=("surface_dice_0_005", "mean"), hd95_norm=("hd95_norm", "mean")).reset_index()
    if not avg.n_seeds.eq(3).all() or set(data.seed) != set(SEEDS) or set(data.method) != set(METHODS):
        raise ValueError("Expected all three methods and seeds")
    data.to_csv(dest/"all_patient_seed_metrics.csv", index=False)
    avg.to_csv(dest/"patient_seed_averages.csv", index=False)
    summary = avg.groupby(["method", "transform"]).agg(n_patients=("dice", "size"), dice_mean=("dice", "mean"),
        native_dice_mean=("native_dice", "mean"), patient_wall_ms_median=("pipeline_wall_ms", "median"),
        mean_regions=("n_regions", "mean")).reset_index()
    summary.to_csv(dest/"method_condition_summary.csv", index=False)
    data.groupby(["method", "seed", "transform"]).agg(n_patients=("dice", "size"), dice_mean=("dice", "mean")).reset_index().to_csv(dest/"per_seed_summary.csv", index=False)
    comparisons = []
    for transform in ("clean", *STRESSES):
        for metric in (["dice", "native_dice"] if transform == "clean" else ["dice"]):
            w = avg[avg["transform"] == transform].pivot(index="patient_id", columns="method", values=metric)
            for a, b in (("local_unet", "coarse_unet"), ("local_combined", "coarse_unet"), ("local_combined", "local_unet")):
                comparisons.append(dict(comparison=f"{a}_minus_{b}", transform=transform, metric=metric, **paired_bootstrap(w[a], w[b])))
    pd.DataFrame(comparisons).to_csv(dest/"paired_descriptive_intervals.csv", index=False)
    clean = data[data["transform"] == "clean"]
    clean.groupby(["method", "seed"]).agg(n_patients=("dice", "size"), native_dice_mean=("native_dice", "mean"),
        min_proposal_gt_coverage=("proposal_gt_pixel_coverage", "min"), min_gt_component_coverage=("proposal_gt_component_coverage_min", "min"),
        mean_proposal_area_fraction=("proposal_area_fraction", "mean"), mean_regions=("n_regions", "mean"),
        empty_predictions=("empty_prediction", "sum")).reset_index().to_csv(dest/"proposal_failure_summary.csv", index=False)
    save_json(dest/"interpretation.json", dict(holdout_used=False, n_validation_patients=int(avg.patient_id.nunique()),
        statistical_unit="patient", aggregation="mean of 3 single-model patient scores; not probability ensemble",
        intervals="10000 paired patient bootstrap repetitions conditional on selected checkpoints; no retraining/selection uncertainty", p_values="not reported",
        timing="decoded RGB to common-grid probability wall time; not comparable to round2 model-only CUDA timings",
        novelty="coarse-to-fine is prior art; proposed task adaptation remains exploratory until validated"))
    print(summary.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser(description="Round3 automatic multi-region refinement; six fits, no install, no holdout inference")
    p.add_argument("--images", default="images")
    p.add_argument("--prepared", default="prepared")
    p.add_argument("--prepared-1024", default="prepared_round2_1024")
    p.add_argument("--round2-output", default="results_round2")
    p.add_argument("--output", default="results_round3")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in selected existing environment; check PYTHON_BIN/GPU visibility. No installation is performed.")
    frame = verify_prepared(args.prepared)
    reference = verify_prepared(args.prepared_1024)
    cols = ["patient_id", "split", "fold", "jpg_sha256", "json_sha256"]
    if not frame[cols].equals(reference[cols]) or read_json(Path(args.prepared)/"audit.json")["side"] != 768 or read_json(Path(args.prepared_1024)/"audit.json")["side"] != 1024:
        raise ValueError("Expected matching frozen 768/1024 prepared data from round2")
    geometry, config = CascadeConfig(), RefinerConfig()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(json.dumps(dict(code_version=__version__, geometry=asdict(geometry), config=asdict(config),
        methods=list(METHODS), seeds=list(SEEDS), n_new_fits=6, fold=0, holdout_used=False,
        manifest_sha256=sha256(Path(args.prepared)/"manifest.csv"), reference_manifest_sha256=sha256(Path(args.prepared_1024)/"manifest.csv"),
        selection_status="post-round2 adaptive exploratory development; not preregistered or independent validation")))
    if (output/"plan.json").exists() and read_json(output/"plan.json") != plan:
        raise ValueError("Round3 plan changed; use new output root, preserve prior results")
    save_json(output/"plan.json", plan)
    save_json(output/"environment.json", dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        environment_installed_or_modified=False))
    # All prerequisites verified BEFORE any new fitting. Final holdout is excluded.
    for seed in SEEDS:
        coarse = Path(args.round2_output)/"resolution_768"/"runs"/"pilot"/"unet"/"fold_0"/f"seed_{seed}"
        coarse_receipt(coarse, args.prepared, frame, seed)
    for row in frame[frame.split == "development"].to_dict("records"):
        source_case(args.images, row)
    frames = []
    for seed in SEEDS:
        coarse = Path(args.round2_output)/"resolution_768"/"runs"/"pilot"/"unet"/"fold_0"/f"seed_{seed}"
        frames.append(diagnose(args.images, args.prepared, args.prepared_1024, frame, output, "coarse_unet", seed, coarse, None, geometry))
        for method in METHODS[1:]:
            run = train_refiner(args.images, args.prepared, args.prepared_1024, frame, output, method, seed, coarse, config, geometry, args.resume)
            frames.append(diagnose(args.images, args.prepared, args.prepared_1024, frame, output, method, seed, coarse, run, geometry))
    summarize(frames, output/"summary")
    from .cli import collect
    archive = output/"round3_results.zip"
    if not archive.exists():
        collect(output, args.prepared, archive)
        import zipfile
        with zipfile.ZipFile(archive, "a", zipfile.ZIP_DEFLATED) as z:
            for name in ("manifest.csv", "audit.json", "split_lock.json", "resolution_receipt.json"):
                z.write(Path(args.prepared_1024)/name, f"prepared_round2_1024/{name}")
    print("Download results_round3/round3_results.zip and logs/06_round3.log; retain all weights/caches/source images on server.", flush=True)


if __name__ == "__main__":
    main()
