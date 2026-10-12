"""No ground-truth crops at inference. All polygons form ONE binary class."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def labelme_mask(annotation, size):
    w, h = size
    if (annotation.get("imageWidth"), annotation.get("imageHeight")) != (w, h):
        raise ValueError(f"JSON/image dimension mismatch: {size}")
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    shapes = annotation.get("shapes", [])
    if not shapes:
        raise ValueError("Empty annotation is not accepted for this positive-only cohort")
    for s in shapes:
        if s.get("label") != "conjunctiva" or s.get("shape_type") != "polygon":
            raise ValueError(f"Unexpected label/type: {s.get('label')}/{s.get('shape_type')}")
        pts = np.asarray(s["points"], dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
            raise ValueError("Invalid polygon coordinates")
        if (pts[:, 0] < 0).any() or (pts[:, 0] > w - 1).any() or (pts[:, 1] < 0).any() or (pts[:, 1] > h - 1).any():
            raise ValueError("Polygon outside image; review manually, never silently clip")
        draw.polygon([tuple(p) for p in pts], fill=255)
    if not np.asarray(mask).any():
        raise ValueError("Zero-area raster mask")
    return mask


def letterbox(image, mask, side=768):
    w, h = image.size
    scale = side / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (side - nw) // 2, (side - nh) // 2
    out = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(image.resize((nw, nh), Image.Resampling.LANCZOS), (left, top))
    m = Image.new("L", (side, side), 0)
    if mask is not None:
        m.paste(mask.resize((nw, nh), Image.Resampling.NEAREST), (left, top))
    valid = np.zeros((side, side), np.float32)
    valid[top:top + nh, left:left + nw] = 1
    return np.asarray(out).copy(), (np.asarray(m) > 0).astype(np.float32), valid, (top, left, nh, nw)


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def load_cached(prepared, row):
    p = Path(prepared) / "cache" / f"{row['patient_id']}.npz"
    with np.load(p, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}
