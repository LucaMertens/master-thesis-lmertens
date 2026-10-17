"""Post-hoc smoothing of the test-side per-cell score map (the fit stays frozen). Unobserved cells stay NaN."""

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.segmentation import slic

SLIC_COMPACTNESS = 10.0
PCA_CHANNELS = 3


def box(scores, observed, size):
    """Mean over a size x size neighbourhood, normalized by the observed cells in it."""
    m = observed.astype(np.float32)
    filled = np.where(observed, np.nan_to_num(scores), 0.0).astype(np.float32)
    num = uniform_filter(filled, size=(1, size, size), mode="constant")
    den = uniform_filter(m, size=(1, size, size), mode="constant")
    out = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)
    return np.where(observed, out, np.nan).astype(np.float32)


def pca_basis(maps, stride=37):
    """(PCA_CHANNELS, C) projection of the BEV features, fitted once and shared across tiers."""
    flat = maps.reshape(-1, maps.shape[-1])[::stride].astype(np.float32)
    _, _, vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
    return vt[:PCA_CHANNELS]


def slic_labels(maps, basis, n_segments):
    """(N, X, Y) superpixel ids computed on the PCA of the BEV features."""
    out = np.empty(maps.shape[:3], dtype=np.int32)
    for i in range(len(maps)):
        rgb = maps[i].astype(np.float32) @ basis.T
        out[i] = slic(rgb, n_segments=n_segments, compactness=SLIC_COMPACTNESS,
                      channel_axis=-1, enforce_connectivity=True)
    return out


def segment_mean(scores, observed, labels):
    """Replace each cell's score by the mean over its segment."""
    out = np.full(scores.shape, np.nan, dtype=np.float32)
    for i in range(len(scores)):
        lab, m = labels[i].ravel(), observed[i].ravel()
        total = np.bincount(lab, weights=np.where(m, np.nan_to_num(scores[i]).ravel(), 0))
        count = np.bincount(lab, weights=m.astype(np.float64))
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        out[i] = np.where(m, mean[lab], np.nan).reshape(scores.shape[1:])
    return out


def apply(method, scores, observed, labels):
    """`none`, `box<k>`, or `slic<n>` given that n's precomputed labels."""
    if method == "none":
        return np.where(observed, scores, np.nan).astype(np.float32)
    if method.startswith("box"):
        return box(scores, observed, int(method[3:]))
    if method.startswith("slic"):
        return segment_mean(scores, observed, labels)
    raise ValueError(method)


def pool(scores, q):
    """(N,) window score: a quantile over the observed cells, NaN where none are."""
    flat = scores.reshape(len(scores), -1)
    with np.errstate(invalid="ignore"):
        return np.nanquantile(flat, q, axis=1)
