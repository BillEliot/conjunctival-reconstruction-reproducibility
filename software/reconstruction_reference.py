"""NumPy reconstruction reference for probabilities already mapped to a canvas.

This executable mathematical example does not reproduce PIL resizing, image
preprocessing, neural-network inference, or model selection. Coordinates are
exclusive integer boxes on the common canvas. The byte-frozen server sources
retain the actual native-to-output mapping and Pillow bilinear interpolation.
"""
import numpy as np


def feather(shape, fraction=.2):
    h, w = shape
    x = np.minimum((np.arange(w)+.5)/w, 1-(np.arange(w)+.5)/w)
    y = np.minimum((np.arange(h)+.5)/h, 1-(np.arange(h)+.5)/h)
    return np.minimum(np.minimum(y[:, None], x[None, :])/fraction, 1).astype(np.float32)


def reconstruct_mapped(coarse, regions, boxes, corrected=False, fraction=.2):
    coarse = np.asarray(coarse, np.float32)
    if coarse.ndim != 2 or not np.isfinite(coarse).all() or np.any((coarse < 0) | (coarse > 1)):
        raise ValueError('Expected finite 2-D coarse probabilities in [0,1]')
    h, w = coarse.shape
    if len(regions) != len(boxes):
        raise ValueError('Each box requires exactly one mapped fine array')
    local = []
    for q, box in zip(regions, boxes):
        l, t, r, b = box
        if not (0 <= l < r <= w and 0 <= t < b <= h):
            raise ValueError('Invalid box')
        q = np.asarray(q, np.float32)
        if q.shape != (b-t, r-l) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError('Expected mapped fine probabilities of the box shape')
        local.append(q)
    if corrected and len(boxes) == 1 and tuple(boxes[0]) == (0, 0, w, h):
        return local[0].copy()
    total, weight = np.zeros_like(coarse), np.zeros_like(coarse)
    for q, (l, t, r, b) in zip(local, boxes):
        alpha = feather(q.shape, fraction)
        total[t:b, l:r] += alpha*q
        weight[t:b, l:r] += alpha
    return (coarse*np.maximum(0, 1-weight)+total)/np.maximum(1, weight)
