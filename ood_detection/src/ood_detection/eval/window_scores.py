"""Every cached window scored by every detector, with its tier."""

import os

import numpy as np

from shared_utils.tiers import assign
from ood_detection.eval.retiered import load_all
from ood_detection.eval.run import load_tier
from ood_detection.report.make_distributions import all_scores

MODES = 3


def window_scores(features_dir, n_modes=MODES):
    """mission, timestamp, tier and one score_<detector> per detector, per window."""
    data = load_all(features_dir)
    train = set(np.unique(load_tier(features_dir, "id_train")["mission"]))
    tier = assign(data["mission"], data["timestamp"], train)
    scores, _ = all_scores(data, tier, modes="kmeans", n_modes=n_modes)
    return {"mission": data["mission"].astype(str), "timestamp": data["timestamp"],
            "tier": np.array([str(t) for t in tier]),
            **{f"score_{k}": np.asarray(v, float) for k, v in scores.items()}}


def cached(features_dir, path=None, n_modes=MODES, rescore=False):
    """window_scores from disk; computed and written on the first call."""
    path = path or os.path.join(features_dir, "eval", "window_scores.npz")
    if os.path.exists(path) and not rescore:
        return dict(np.load(path, allow_pickle=True))
    out = window_scores(features_dir, n_modes)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)
    print(f"scored {len(out['tier'])} windows -> {path}")
    return out
