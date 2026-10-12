"""Patient/photo-level binary semantic-segmentation metrics.

Call with the aspect-preserved, resized *valid image rectangle*, after cropping
letterbox padding. Both eyes and every annotated polygon form one foreground;
never keep only the largest connected component. One photo is one patient.

Dice is the prespecified primary metric, averaged across patients rather than
pooled pixels. Complementary overlap and boundary metrics follow the problem-
aware approach of Metrics Reloaded (Maier-Hein et al., Nature Methods, 2024,
https://doi.org/10.1038/s41592-023-02151-z). This module does not establish
clinical utility, independent test validity, or inter-observer reproducibility.

Boundary metrics use foreground boundary pixel centres (4-connected erosion)
and Euclidean distance transforms, in resized pixels. Surface Dice is the
fraction of boundary pixels within 0.005 or 0.01 of the valid rectangle's
diagonal. It is a pixel-count approximation, not a subpixel/arc-length weighted
surface calculation. These tolerances are technical sensitivity analyses, not
clinically established distances. HD95 is the maximum of the two directional
95th percentiles; ASSD pools both directions with equal weight per boundary
pixel. Both distances are divided by sqrt(height**2 + width**2).

Both-empty masks score 1 for Dice/IoU/surface Dice and 0 for distances. One-empty
masks score 0 for Dice/IoU/surface Dice and receive a finite distance failure
penalty of one image diagonal (normalised value 1), explicitly flagged. Precision
is 1 if no pixels are predicted; recall is 1 if no GT pixels exist (vacuous
conventions). Empty GT is not expected in the positive-only research cohort and
must be reviewed, rather than silently treated as routine evidence.

8-connected GT-component coverage and absolute component-count error are
secondary semantic-mask diagnostics only. Components are not anatomical eye
instances; fragments/bridges can change their count.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy import ndimage


METRIC_NAMES = (
    "dice", "iou", "precision", "recall", "surface_dice_0_005",
    "surface_dice_0_01", "hd95_norm", "assd_norm",
    "gt_component_recall_0_5", "component_count_error",
)


def _binary_2d(value: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.ndim != 2 or min(mask.shape, default=0) == 0:
        raise ValueError(f"{name} must be a non-empty 2-D binary mask")
    if not np.isfinite(mask).all() or not np.logical_or(mask == 0, mask == 1).all():
        raise ValueError(f"{name} must contain only binary 0/1 values; threshold probabilities first")
    return mask.astype(bool, copy=False)


def _boundary(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(
        mask, structure=ndimage.generate_binary_structure(2, 1), border_value=0
    )
    return mask & ~eroded


def crop_valid_region(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    """Remove a rectangular letterbox validity mask and return binary arrays.

    Validity must be one non-empty solid rectangle. Holes or multiple regions
    are rejected because normalised boundary distances need a defined image
    diagonal, not the padded square. Predictions outside it never contribute.
    """
    pred, gt, valid = (_binary_2d(x, n) for x, n in
                       ((pred, "pred"), (gt, "gt"), (valid, "valid")))
    if pred.shape != gt.shape or pred.shape != valid.shape:
        raise ValueError("pred, gt and valid must have the same shape")
    coords = np.argwhere(valid)
    if not coords.size:
        raise ValueError("valid mask must contain a non-empty rectangle")
    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0) + 1
    if valid.sum() != (bottom - top) * (right - left):
        raise ValueError("valid mask must be a solid rectangle")
    sl = (slice(top, bottom), slice(left, right))
    return pred[sl], gt[sl]


def compute_metrics(pred_bool: np.ndarray, gt_bool: np.ndarray) -> Dict[str, float]:
    """Evaluate all foreground pixels without component-removal postprocessing."""
    pred = _binary_2d(pred_bool, "pred")
    gt = _binary_2d(gt_bool, "gt")
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} != {gt.shape}")

    pred_n, gt_n = int(pred.sum()), int(gt.sum())
    intersection = int(np.logical_and(pred, gt).sum())
    union = pred_n + gt_n - intersection
    result = {
        "dice": 2.0 * intersection / (pred_n + gt_n) if pred_n + gt_n else 1.0,
        "iou": intersection / union if union else 1.0,
        "precision": intersection / pred_n if pred_n else 1.0,
        "recall": intersection / gt_n if gt_n else 1.0,
        "empty_prediction": int(pred_n == 0),
        "empty_ground_truth": int(gt_n == 0),
        "empty_distance_penalty": int((pred_n == 0) != (gt_n == 0)),
    }
    diagonal = math.hypot(*pred.shape)
    if not pred_n or not gt_n:
        both_empty = not pred_n and not gt_n
        result.update({
            "surface_dice_0_005": float(both_empty),
            "surface_dice_0_01": float(both_empty),
            "hd95_norm": 0.0 if both_empty else 1.0,
            "assd_norm": 0.0 if both_empty else 1.0,
        })
    else:
        pred_surface, gt_surface = _boundary(pred), _boundary(gt)
        pred_to_gt = ndimage.distance_transform_edt(~gt_surface)[pred_surface]
        gt_to_pred = ndimage.distance_transform_edt(~pred_surface)[gt_surface]
        distances = np.concatenate((pred_to_gt, gt_to_pred))
        result.update({
            "surface_dice_0_005": float(np.mean(distances <= 0.005 * diagonal)),
            "surface_dice_0_01": float(np.mean(distances <= 0.01 * diagonal)),
            "hd95_norm": float(max(np.percentile(pred_to_gt, 95),
                                     np.percentile(gt_to_pred, 95)) / diagonal),
            "assd_norm": float(distances.mean() / diagonal),
        })

    structure = ndimage.generate_binary_structure(2, 2)
    gt_labels, gt_components = ndimage.label(gt, structure=structure)
    _, pred_components = ndimage.label(pred, structure=structure)
    covered = 0
    if gt_components:
        sizes = np.bincount(gt_labels.ravel(), minlength=gt_components + 1)[1:]
        overlaps = np.bincount(gt_labels[pred], minlength=gt_components + 1)[1:]
        covered = int(np.count_nonzero(overlaps / sizes >= 0.5))
    result.update({
        "gt_component_recall_0_5": covered / gt_components if gt_components else 1.0,
        "component_count_error": abs(int(pred_components) - int(gt_components)),
        "gt_components": int(gt_components),
        "pred_components": int(pred_components),
        "foreground_gt_pixels": gt_n,
        "foreground_pred_pixels": pred_n,
    })
    return result
