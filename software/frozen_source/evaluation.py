"""Probability ensembles, prespecified photometric stress tests, no GT-informed ROI."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from .audit import save_json, sha256, verify_prepared
from .data import load_cached, read_json
from .models import Segmenter
from .metrics import compute_metrics

STRESSES = ("dark", "bright", "gamma_low", "gamma_high", "warm", "cool")


def stress(rgb, name):
    if name not in ("clean", *STRESSES):
        raise ValueError(f"Unknown photometric stress: {name}")
    if name == "clean": return rgb
    if name == "dark": return (rgb * .7).clamp(0, 1)
    if name == "bright": return (rgb * 1.3).clamp(0, 1)
    if name == "gamma_low": return rgb.clamp_min(0).pow(.7)
    if name == "gamma_high": return rgb.clamp_min(0).pow(1.5)
    gain = [.15, 0, -.15] if name == "warm" else [-.15, 0, .15]
    return (rgb * (1 + torch.tensor(gain, device=rgb.device).view(1,3,1,1))).clamp(0,1)


def crop(a, box):
    t, l, h, w = map(int, box)
    return a[t:t+h, l:l+w]


def preview(a, probability, name, path):
    rgb = a["image"].copy(); valid = a["valid"] > 0
    gt = (a["mask"] > 0) & valid; pr = (probability >= .5) & valid
    ge = gt ^ ndi.binary_erosion(gt); pe = pr ^ ndi.binary_erosion(pr)
    rgb[ndi.binary_dilation(ge)] = (0,255,70)
    rgb[ndi.binary_dilation(pe)] = (255,70,60)
    im = Image.fromarray(rgb).resize((512,512))
    ImageDraw.Draw(im).text((8,8), f"{name}: GT green / pred red", fill="white", stroke_width=1, stroke_fill="black")
    im.save(path,quality=90)


@torch.inference_mode()
def evaluate_checkpoint(model, prepared, frame, output, device, batch_size=6, reference_prepared=None, save_probabilities=True):
    out = Path(output); (out / "val_predictions").mkdir(exist_ok=True)
    (out / "val_overlays").mkdir(exist_ok=True)
    model.eval(); rows=[]
    for row in frame.to_dict("records"):
        a=load_cached(prepared,row)
        x=torch.from_numpy(a["image"].transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/255
        prob=model(x)[0].sigmoid()[0,0].cpu().numpy()
        if save_probabilities:
            np.save(out / "val_predictions" / f"{row['patient_id']}.npy",prob.astype(np.float32))
        if reference_prepared is None:
            met=compute_metrics(crop(prob>=.5,a["box"]),crop(a["mask"]>0,a["box"]))
        else:
            ref = load_cached(reference_prepared, row)
            gt = crop(ref["mask"] > 0, ref["box"])
            resized = np.asarray(Image.fromarray(crop(prob, a["box"])).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR))
            met = compute_metrics(resized >= .5, gt)
            met.update(reference_side=read_json(Path(reference_prepared)/"audit.json")["side"], evaluation_grid="common valid-content grid")
        met.update(patient_id=row["patient_id"],n_polygons=row["n_polygons"],has_dng=row["has_dng"])
        rows.append(met)
        preview(a,prob,row["patient_id"],out / "val_overlays" / f"{row['patient_id']}.jpg")
    pd.DataFrame(rows).to_csv(out / "val_metrics.csv",index=False)


def freeze(prepared, output, families, folds, seeds):
    df=verify_prepared(prepared); output=Path(output)
    if (output / "freeze.json").exists():
        raise RuntimeError("freeze.json already exists. Preserve this test commitment; use a new output for an intentionally new study.")
    entries={}; config_hashes=set()
    for family in families:
        entries[family]=[]
        for fold in folds:
            for seed in seeds:
                run=output / "runs" / "development" / family / f"fold_{fold}" / f"seed_{seed}"
                done=read_json(run / "completed.json")
                ck=run / "best.pt"
                if sha256(ck)!=done["best_sha256"]:
                    raise ValueError(f"Checkpoint modified: {ck}")
                ident=done["identity"]
                if ident["manifest_sha256"]!=sha256(Path(prepared)/"manifest.csv"):
                    raise ValueError("Dataset receipt mismatch")
                config_hashes.add(ident["config_sha256"])
                entries[family].append(dict(path=str(ck.relative_to(output)),sha256=sha256(ck),identity=ident))
    if len(config_hashes)!=1:
        raise ValueError("Formal models must share the resolved training config")
    if sorted(folds)!=list(range(5)) or len(seeds)!=3:
        raise ValueError("Formal locked analysis prespecifies 5 folds and 3 seeds for each family")
    receipt=dict(schema_version=1,manifest_sha256=sha256(Path(prepared)/"manifest.csv"),aggregation="probability_mean",threshold=.5,seeds=list(seeds),folds=list(folds),families=list(families),checkpoints=entries,holdout_patient_ids=sorted(df[df.split=="holdout"].patient_id.tolist()),primary_comparison="proposed vs unet",stresses=list(STRESSES),evaluation_grid="letterboxed cache cropped to valid content",selection_basis="development only; user explicitly runs freeze after reviewing development")
    save_json(output / "freeze.json",receipt)
    print(f"Freeze receipt: {output/'freeze.json'}",flush=True)


@torch.inference_mode()
def final_evaluation(prepared, output, release_test=False):
    if not release_test:
        raise RuntimeError("Holdout remains locked. After development review run freeze, then explicitly use --release-test.")
    if not torch.cuda.is_available():
        raise RuntimeError("Final ensemble inference requires CUDA")
    output=Path(output); df=verify_prepared(prepared)
    fr=read_json(output / "freeze.json")
    if fr["manifest_sha256"]!=sha256(Path(prepared)/"manifest.csv"):
        raise ValueError("Freeze/data hash mismatch")
    cases=df[df.split=="holdout"].to_dict("records")
    if sorted(r["patient_id"] for r in cases)!=fr["holdout_patient_ids"]:
        raise ValueError("Holdout patient IDs changed")
    device=torch.device("cuda")
    for family in fr["families"]:
        dest=output / "final" / family
        if (dest / "metadata.json").exists():
            raise RuntimeError(f"Final result already exists: {dest}; do not repeatedly retune from holdout.")
        if dest.exists():
            raise RuntimeError(f"Partial final results exist: {dest}. Review the failure before restarting in a documented new output root.")
        dest.mkdir(parents=True)
        for sub in ("probabilities","overlays","native_masks"):
            (dest/sub).mkdir()
        # CPU probability accumulation: one GPU model at a time, about 1 GiB for 54 cases.
        sums={}; squares={}; corrupt={}
        t_start=time.time(); latencies=[]; n=len(fr["checkpoints"][family])
        for j, entry in enumerate(fr["checkpoints"][family]):
            p=output/entry["path"]
            if sha256(p)!=entry["sha256"]: raise ValueError(f"Checkpoint hash mismatch: {p}")
            ck=torch.load(p,map_location="cpu",weights_only=False)
            if ck["identity"]!=entry["identity"]: raise ValueError("Checkpoint identity changed")
            model=Segmenter(family,False).to(device).eval(); model.load_state_dict(ck["model"])
            for k, row in enumerate(cases):
                a=load_cached(prepared,row); pid=row["patient_id"]
                x=torch.from_numpy(a["image"].transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/255
                if j==0:
                    sums[pid]=np.zeros(a["mask"].shape,np.float32)
                    squares[pid]=np.zeros(a["mask"].shape,np.float32)
                    corrupt[pid]={s:np.zeros(a["mask"].shape,np.float32) for s in STRESSES}
                # Fixed warmup, synchronized latency, batch=1, excludes I/O and CPU metric cost.
                if k==0:
                    for _ in range(5): model(x)
                torch.cuda.synchronize(); t0=time.perf_counter()
                probability=model(x)[0].sigmoid()[0,0].cpu().numpy()
                torch.cuda.synchronize(); latencies.append(time.perf_counter()-t0)
                sums[pid]+=probability; squares[pid]+=probability**2
                for s in STRESSES:
                    ps=model(stress(x,s))[0].sigmoid()[0,0].cpu().numpy()
                    corrupt[pid][s]+=ps
            del model; torch.cuda.empty_cache()
            print(f"{family}: ensemble member {j+1}/{n}",flush=True)
        rows=[]; stress_rows=[]
        for row in cases:
            pid=row["patient_id"]; a=load_cached(prepared,row)
            probability=sums[pid]/n
            uncertainty=np.sqrt(np.maximum(squares[pid]/n-probability**2,0))
            gt=crop(a["mask"]>0,a["box"])
            met=compute_metrics(crop(probability>=fr["threshold"],a["box"]),gt)
            stress_dice=[]
            for s in STRESSES:
                cm=compute_metrics(crop(corrupt[pid][s]/n>=.5,a["box"]),gt)
                stress_dice.append(cm["dice"])
                stress_rows.append(dict(patient_id=pid,stress=s,**cm))
            # Ensemble dispersion is not a calibrated uncertainty estimate.
            band=(crop(probability,a["box"])>=.1)&(crop(probability,a["box"])<=.9)
            unc=crop(uncertainty,a["box"])
            met.update(patient_id=pid,family=family,n_polygons=row["n_polygons"],has_dng=row["has_dng"],foreground_fraction=row["foreground_fraction"],corrupt_dice_mean=float(np.mean(stress_dice)),uncertainty_mean=float(unc[band].mean()) if band.any() else 0.,aggregation="probability_mean",n_checkpoints=n)
            rows.append(met)
            np.savez_compressed(dest/"probabilities"/f"{pid}.npz",probability=probability.astype(np.float16),ensemble_std=uncertainty.astype(np.float16),box=a["box"])
            # For downstream color sampling: native-sized predicted mask, with no access to native GT.
            native=Image.fromarray((crop(probability>=.5,a["box"])*255).astype(np.uint8)).resize((int(row["width"]),int(row["height"])),Image.Resampling.NEAREST)
            native.save(dest/"native_masks"/f"{pid}.png")
            preview(a,probability,pid,dest/"overlays"/f"{pid}.jpg")
        pd.DataFrame(rows).to_csv(dest/"test_metrics.csv",index=False)
        pd.DataFrame(stress_rows).to_csv(dest/"stress_metrics.csv",index=False)
        save_json(dest/"metadata.json",dict(family=family,aggregation="probability_mean",n_checkpoints=n,holdout_patient_ids=fr["holdout_patient_ids"],seeds=fr["seeds"],folds=fr["folds"],manifest_sha256=fr["manifest_sha256"],freeze_sha256=sha256(output/"freeze.json"),threshold=fr["threshold"],n_patients=len(cases),evaluation_grid=fr["evaluation_grid"],single_member_mean_latency_ms=float(np.mean(latencies)*1000),ensemble_sequential_estimated_latency_ms=float(np.mean(latencies)*n*1000),elapsed_seconds=time.time()-t_start,gpu=torch.cuda.get_device_name(0),failure_dice_threshold=.5,uncertainty="uncalibrated probability standard deviation across checkpoints"))
        del sums,squares,corrupt
    print("Holdout evaluated. Preserve all outputs and report every frozen family.",flush=True)
