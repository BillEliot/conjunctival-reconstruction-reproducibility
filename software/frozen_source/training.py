"""Development only. Holdout rows are never passed to these datasets."""
import csv
import hashlib
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
from .audit import save_json, sha256, verify_prepared
from .data import load_cached, read_json
from .models import Segmenter


@dataclass
class TrainConfig:
    epochs: int = 120
    patience: int = 20
    batch_size: int = 6
    workers: int = 4
    lr: float = 0.0003
    weight_decay: float = 0.0001
    boundary_weight: float = 0.2
    consistency_weight: float = 0.5
    consistency_ramp_epochs: int = 15
    pretrained: bool = True
    side: int = 768


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Deterministic selection/seeds; CUDA may still differ across drivers/devices.
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_seed(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed); random.seed(seed)


class Cases(Dataset):
    def __init__(self, prepared, frame, train=False, memory=None):
        self.prepared = prepared
        self.rows = frame.to_dict("records")
        self.train = train
        self.memory = memory
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]; a = load_cached(self.prepared, row) if self.memory is None else self.memory.get(row)
        rgb = a["image"].astype(np.float32)/255
        m, valid = a["mask"].astype(np.float32), a["valid"].astype(np.float32)
        flipped = self.train and random.random() < .5
        if flipped:
            rgb = rgb[:, ::-1].copy(); m = m[:, ::-1].copy(); valid = valid[:, ::-1].copy()
        # Narrow boundary band, clipped to valid pixels; no forced connectivity.
        edge = ((ndi.binary_dilation(m > 0, iterations=2) ^ ndi.binary_erosion(m > 0, iterations=2)).astype(np.float32) * valid if self.memory is None else
                (a["edge"][:, ::-1].astype(np.float32) if flipped else a["edge"].astype(np.float32)))
        return dict(image=torch.from_numpy(rgb.transpose(2, 0, 1).copy()), mask=torch.from_numpy(m[None]), valid=torch.from_numpy(valid[None]), edge=torch.from_numpy(edge[None]), patient_id=row["patient_id"])


def photometric(x, valid):
    """Same anatomy/geometric transform, different intensity/color rendering."""
    b = x.shape[0]
    gain = .85 + torch.rand((b, 3, 1, 1), device=x.device) * .3
    gamma = .8 + torch.rand((b, 1, 1, 1), device=x.device) * .4
    brightness = .85 + torch.rand((b, 1, 1, 1), device=x.device) * .3
    y = x.clamp(1e-6, 1).pow(gamma) * gain * brightness
    saturation = .8 + torch.rand((b, 1, 1, 1), device=x.device) * .4
    gray = y.mean(1, keepdim=True)
    y = gray + saturation * (y - gray)
    y = (y + torch.randn_like(y) * .005).clamp(0, 1)
    return y * valid


def masked_mean(x, valid):
    return (x * valid).sum() / valid.sum().clamp_min(1)


def seg_loss(logits, target, valid):
    bce = masked_mean(nn.functional.binary_cross_entropy_with_logits(logits.float(), target, reduction="none"), valid)
    p = logits.float().sigmoid() * valid
    t = target * valid
    dice = (2 * (p*t).sum((1, 2, 3)) + 1)/(p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
    return .5 * bce + .5 * (1 - dice.mean())


def edge_loss(logits, edge, valid):
    return seg_loss(logits, edge, valid)


@torch.inference_mode()
def validation(model, loader, device):
    model.eval(); ds = []
    for batch in loader:
        x, m, v = [batch[k].to(device) for k in ("image", "mask", "valid")]
        pred = (model(x)[0].sigmoid() >= .5).float() * v
        dice = (2 * (pred*m).sum((1,2,3)) + 1e-8)/(pred.sum((1,2,3)) + (m*v).sum((1,2,3)) + 1e-8)
        ds.extend(dice.cpu().tolist())
    return float(np.mean(ds))


@torch.inference_mode()
def validation_common(model, loader, reference_prepared, frame, device, reference_memory=None):
    """Select checkpoints on one reference grid; crop padding BEFORE resize."""
    from .evaluation import crop
    model.eval(); scores = []; lookup = frame.set_index("patient_id")
    for batch in loader:
        probs = model(batch["image"].to(device))[0].sigmoid().float().cpu().numpy()[:, 0]
        for probability, pid in zip(probs, batch["patient_id"]):
            row = lookup.loc[pid].to_dict(); row["patient_id"] = pid
            ref = load_cached(reference_prepared, row) if reference_memory is None else reference_memory.get(row)
            gt = crop(ref["mask"] > 0, ref["box"])
            box = tuple(int(row[k]) for k in ("box_top", "box_left", "box_height", "box_width"))
            p = np.asarray(Image.fromarray(crop(probability, box)).resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR)) >= .5
            scores.append(float(2*np.count_nonzero(p & gt)/max(1, np.count_nonzero(p)+np.count_nonzero(gt))))
    return float(np.mean(scores))


def train_one(prepared, output, family, fold, seed, cfg, stage="development", resume=False, reference_prepared=None, save_validation_probabilities=True, runtime=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable in the current Python process. Check PYTHON_BIN, the active server environment, and GPU visibility. No environment installation is performed by this program.")
    memory = runtime.store(prepared) if runtime is not None else None
    df = verify_prepared(prepared) if memory is None else memory.verify()
    side = read_json(Path(prepared) / "audit.json")["side"]
    if side != cfg.side:
        raise ValueError("Config side differs from prepared cache; do not silently rescale")
    tr = df[(df.split == "development") & (df.fold != fold)]
    va = df[(df.split == "development") & (df.fold == fold)]
    if tr.empty or va.empty:
        raise ValueError("Empty development fold")
    out = Path(output) / "runs" / stage / family / f"fold_{fold}" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    state_path, best_path = out / "last.pt", out / "best.pt"
    identity = dict(family=family, fold=fold, seed=seed, config=asdict(cfg), config_sha256=cfg_hash(cfg), manifest_sha256=sha256(Path(prepared)/"manifest.csv"), stage=stage)
    if reference_prepared is not None:
        ref = verify_prepared(reference_prepared) if runtime is None else runtime.store(reference_prepared).verify()
        keys = ["patient_id", "split", "fold", "jpg_sha256", "json_sha256"]
        if not ref[keys].equals(df[keys]):
            raise ValueError("Reference cache differs in source or patient split")
        identity.update(reference_manifest_sha256=sha256(Path(reference_prepared)/"manifest.csv"),
                        validation_reference_side=read_json(Path(reference_prepared)/"audit.json")["side"],
                        save_validation_probabilities=bool(save_validation_probabilities))
    if (out / "completed.json").exists():
        old = read_json(out / "completed.json")
        if old["identity"] != identity or sha256(best_path) != old["best_sha256"]:
            raise ValueError("Completed run configuration or weight hash changed; preserve outputs and review")
        print(f"Verified completed run: {out}", flush=True)
        return out
    if state_path.exists() and not resume:
        raise RuntimeError(f"Unfinished run exists at {out}; explicitly pass --resume or use new output")
    seed_all(seed)
    device = torch.device("cuda")
    model = Segmenter(family, cfg.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda")
    start, best, stale, history = 0, -1., 0, []
    history_path = out / "history.csv"
    if resume and state_path.exists():
        ck = torch.load(state_path, map_location="cpu", weights_only=False)
        if ck["identity"] != identity:
            raise ValueError("Resume receipt differs from dataset/configuration")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start, best, stale = ck["epoch"]+1, ck["best"], ck["stale"]
        history = ck["history"] if "history" in ck else (pd.read_csv(history_path).to_dict("records") if history_path.exists() else [])
        history = [r for r in history if int(r["epoch"]) <= start]
        random.setstate(ck["python_rng"]); np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state_all(ck["cuda_rng"])
    save_json(out / "identity.json", identity)
    save_json(out / "environment.json", dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), gpu_bytes=torch.cuda.get_device_properties(0).total_memory))
    val_loader = DataLoader(Cases(prepared, va, memory=memory), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers if runtime is None else 0, pin_memory=True, worker_init_fn=worker_seed)
    dataset = Cases(prepared, tr, True, memory=memory)
    from .performance import CheckpointWriter, timed_batches, write_profile
    writer = CheckpointWriter(out) if runtime is not None else None
    next_epoch = None
    try:
        for epoch in range(start, cfg.epochs):
            if stale >= cfg.patience:
                break
            # Epoch-bound generator allows reproducible shuffle after restarting.
            generator = torch.Generator().manual_seed(seed * 10000 + epoch)
            if next_epoch is None:
                loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, worker_init_fn=worker_seed, generator=generator, **(runtime.loader_options(cfg.workers) if runtime is not None else {}))
                batches = iter(loader)
            else:
                loader, batches = next_epoch; next_epoch = None
            model.train(); losses=[]; t0=time.time(); profile = dict(data_wait_seconds=0.)
            for batch in tqdm(timed_batches(batches, profile), total=len(loader), desc=f"{stage}/{family}/f{fold}/s{seed}/e{epoch+1}"):
                x, m, v, edge = [batch[k].to(device, non_blocking=True) for k in ("image","mask","valid","edge")]
                x2 = photometric(x, v)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    l1, e1 = model(x); l2, e2 = model(x2)
                    # EVERY architecture receives the same two supervised views.
                    loss = .5 * (seg_loss(l1, m, v) + seg_loss(l2, m, v))
                    if model.use_boundary:
                        loss = loss + cfg.boundary_weight * .5 * (edge_loss(e1, edge, v) + edge_loss(e2, edge, v))
                    if model.use_consistency:
                        ramp = min(1., (epoch+1)/cfg.consistency_ramp_epochs)
                        region_weight = v * (1 + 4*m + 2*edge)
                        consistency = masked_mean((l2.float().sigmoid()-l1.float().sigmoid().detach()).square(), region_weight)
                        loss = loss + cfg.consistency_weight * ramp * consistency
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss; see logs and fix explicitly")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer); scaler.update()
                losses.append(float(loss.detach()))
            train_seconds = time.time()-t0; val_start = time.perf_counter()
            score = (validation(model, val_loader, device) if reference_prepared is None else
                     validation_common(model, val_loader, reference_prepared, va, device, runtime.store(reference_prepared) if runtime is not None else None))
            validation_seconds = time.perf_counter()-val_start
            if not np.isfinite(score):
                raise FloatingPointError("Non-finite validation score")
            scheduler.step()
            better = score > best
            if better:
                best, stale = score, 0
                if writer is None:
                    torch.save(dict(model=model.state_dict(), identity=identity, epoch=epoch, val_dice=score), best_path)
            else:
                stale += 1
            row=dict(epoch=epoch+1, train_loss=float(np.mean(losses)), val_dice=score, best_dice=best, seconds=time.time()-t0, lr=optimizer.param_groups[0]["lr"])
            history.append(row)
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), identity=identity, epoch=epoch, best=best, stale=stale, history=history, python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            if writer is None:
                tmp = out / "last.tmp.pt"; torch.save(checkpoint, tmp); os.replace(tmp, state_path)
            else:
                writer.wait()  # No active writer while Linux workers are forked.
                if stale < cfg.patience and epoch+1 < cfg.epochs:
                    generator_next = torch.Generator().manual_seed(seed*10000+epoch+1)
                    loader_next = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, worker_init_fn=worker_seed, generator=generator_next, **runtime.loader_options(cfg.workers))
                    next_epoch = (loader_next, iter(loader_next))
                writer.submit(checkpoint, better)
                write_profile(out, epoch, profile, train_seconds, validation_seconds)
            pd.DataFrame(history).to_csv(history_path, index=False)
            print(json.dumps(row), flush=True)
    finally:
        next_epoch = None
        if writer is not None:
            writer.close()
    pd.DataFrame(history).to_csv(history_path, index=False)
    ck=torch.load(best_path,map_location="cpu",weights_only=False)
    model.load_state_dict(ck["model"])
    from .evaluation import evaluate_checkpoint
    evaluate_checkpoint(model, prepared, va, out, device, cfg.batch_size, reference_prepared, save_validation_probabilities)
    save_json(out / "completed.json",dict(identity=identity,best_epoch=ck["epoch"]+1,best_val_dice=ck["val_dice"],best_sha256=sha256(best_path)))
    del model, optimizer; torch.cuda.empty_cache()
    return out
