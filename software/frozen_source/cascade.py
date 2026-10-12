"""Automatic multi-region inference. This module never reads reference annotations."""
from dataclasses import asdict, dataclass
import math
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from .data import letterbox
from .evaluation import crop, stress


@dataclass(frozen=True)
class CascadeConfig:
    coarse_side: int = 768
    patch_side: int = 512
    reference_side: int = 1024
    proposal_threshold: float = .25
    context_factor: float = 2.5
    minimum_native_side: int = 128
    max_regions: int = 8
    feather_fraction: float = .2
    output_threshold: float = .5

    def __post_init__(self):
        if any(s < 32 or s % 32 for s in (self.coarse_side, self.patch_side, self.reference_side)):
            raise ValueError("Model/reference sides must be positive multiples of 32")
        if not 0 < self.proposal_threshold < 1 or not 0 < self.output_threshold < 1:
            raise ValueError("Invalid probability threshold")
        if self.context_factor < 1 or self.minimum_native_side < 1 or self.max_regions < 1 or not 0 < self.feather_fraction <= .5:
            raise ValueError("Invalid proposal/fusion geometry")


def context_box(box, size, factor=2.5, minimum=128, jitter=(0., 0.)):
    """Square context around an exclusive (left,top,right,bottom) box; clip to image."""
    w, h = map(int, size)
    l, t, r, b = map(float, box)
    if not (w > 0 and h > 0 and 0 <= l < r <= w and 0 <= t < b <= h):
        raise ValueError("Invalid native box")
    side = max(minimum, factor * max(r-l, b-t))
    cx, cy = (l+r)/2 + jitter[0]*(r-l), (t+b)/2 + jitter[1]*(b-t)
    return (max(0, math.floor(cx-side/2)), max(0, math.floor(cy-side/2)),
            min(w, math.ceil(cx+side/2)), min(h, math.ceil(cy+side/2)))


def propose_regions(coarse_content, native_size, config=CascadeConfig()):
    """Use only coarse valid-content probabilities. Preserve ALL components.

    Overflow uses the union bounding box of every candidate, rather than truncating
    a ranking or retaining the largest component. Empty prediction uses full frame.
    No component area filter or left/right anatomical assumption is applied.
    """
    p = np.asarray(coarse_content)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Expected finite 2D probabilities in [0,1]")
    w, h = map(int, native_size)
    if min(w, h) < 1:
        raise ValueError("Invalid native dimensions")
    labels, n = ndi.label(p >= config.proposal_threshold, structure=np.ones((3, 3)))
    if n == 0:
        return [(0, 0, w, h)], {"raw_component_count": 0, "proposal_mode": "empty_full_frame"}
    boxes = []
    for sy, sx in ndi.find_objects(labels):
        boxes.append((math.floor(sx.start*w/p.shape[1]), math.floor(sy.start*h/p.shape[0]),
                      min(w, math.ceil(sx.stop*w/p.shape[1])), min(h, math.ceil(sy.stop*h/p.shape[0]))))
    mode = "all_components"
    if n > config.max_regions:
        boxes = [(min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))]
        mode = "overflow_union_all"
    return [context_box(b, (w, h), config.context_factor, config.minimum_native_side) for b in boxes], {"raw_component_count": n, "proposal_mode": mode}


def resize_probability(prob, size):
    return np.asarray(Image.fromarray(np.asarray(prob, np.float32)).resize(tuple(map(int, size)), Image.Resampling.BILINEAR))


def feather(shape, fraction):
    h, w = shape
    x = np.minimum((np.arange(w)+.5)/w, 1-(np.arange(w)+.5)/w)
    y = np.minimum((np.arange(h)+.5)/h, 1-(np.arange(h)+.5)/h)
    return np.minimum(np.minimum(y[:, None], x[None, :])/fraction, 1).astype(np.float32)


def fuse_probabilities(coarse_content, region_probabilities, boxes, native_size, output_size, config=CascadeConfig()):
    """Feather average overlapping predictions; retain coarse output outside ROIs."""
    if len(region_probabilities) != len(boxes):
        raise ValueError("Region probabilities and boxes do not match")
    w, h = map(int, output_size)
    nw, nh = map(int, native_size)
    coarse = resize_probability(coarse_content, (w, h))
    weight, total = np.zeros_like(coarse), np.zeros_like(coarse)
    for p, box in zip(region_probabilities, boxes):
        l, t, r, b = box
        if not (0 <= l < r <= nw and 0 <= t < b <= nh):
            raise ValueError("Invalid fusion box")
        l, t, r, b = math.floor(l*w/nw), math.floor(t*h/nh), min(w, math.ceil(r*w/nw)), min(h, math.ceil(b*h/nh))
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Invalid local probability")
        local = resize_probability(p, (r-l, b-t))
        alpha = feather(local.shape, config.feather_fraction)
        total[t:b, l:r] += alpha * local
        weight[t:b, l:r] += alpha
    return (coarse*np.maximum(0, 1-weight) + total) / np.maximum(1, weight)


def image_tensor(image, side, device, transform="clean"):
    rgb, _, valid, box = letterbox(image, None, side)
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None].to(device)/255
    v = torch.from_numpy(valid)[None, None].to(device)
    return stress(x, transform)*v, box


@torch.inference_mode()
def predict_regions(image, coarse_model, refiner, device, config=CascadeConfig(), transform="clean"):
    """Decoded RGB -> coarse prediction -> automatic boxes -> local probabilities.

    Photometric transforms are applied after each image/patch resize, identically
    to the two model inputs. They are fixed synthetic conditions, not new patients.
    No reference-mask argument is accepted anywhere on this inference path.
    """
    x, box = image_tensor(image, config.coarse_side, device, transform)
    p = coarse_model(x)[0].sigmoid()[0, 0].float().cpu().numpy()
    content = crop(p, box)
    boxes, meta = (propose_regions(content, image.size, config) if refiner is not None else
                   ([], {"raw_component_count": 0, "proposal_mode": "coarse_only"}))
    patches = []
    if refiner is not None:
        for bounds in boxes:
            x, patch_box = image_tensor(image.crop(bounds), config.patch_side, device, transform)
            prob = refiner(x)[0].sigmoid()[0, 0].float().cpu().numpy()
            patches.append(crop(prob, patch_box))
    return {"coarse_content": content, "boxes": boxes, "patch_probabilities": patches,
            "native_size": image.size, **meta}


def reconstruct(prediction, output_size, config=CascadeConfig()):
    if not prediction["patch_probabilities"]:
        return resize_probability(prediction["coarse_content"], output_size)
    return fuse_probabilities(prediction["coarse_content"], prediction["patch_probabilities"],
        prediction["boxes"], prediction["native_size"], output_size, config)
