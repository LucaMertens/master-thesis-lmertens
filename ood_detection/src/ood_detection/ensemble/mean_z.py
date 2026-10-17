"""Mean of the members' z-scores, standardized on ID training windows with equal weights (no tier labels used)."""

import numpy as np

# One detector per failure mode: visual, geometric, silent hazards
TRIO = ("image_mahalanobis", "viewpoint_range_med", "depth_entropy")
ENSEMBLE = "ensemble"


def zscore(s, fit):
    v = s[fit & np.isfinite(s)]
    return (s - v.mean()) / v.std()


def mean_z(scores, fit, keys=TRIO):
    """(N,) window score, NaN wherever a member has none."""
    z = np.stack([zscore(np.asarray(scores[k], float), fit) for k in keys])
    ok = np.isfinite(z).all(0)
    out = np.full(z.shape[1], np.nan)
    out[ok] = z[:, ok].mean(0)
    return out
