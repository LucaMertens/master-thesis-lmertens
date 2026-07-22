"""Synthetic features dir mimicking ood_detection.features.extract output."""

import numpy as np
import pytest

BEV_DIM, DINO_DIM = 8, 16


def make_tier(rng, missions, n_per=120, feat_shift=0.0, err_scale=0.05, hz=1.0):
    n = n_per * len(missions)
    mission = np.repeat(np.array(missions), n_per)
    timestamp = np.tile(np.arange(n_per) / hz, len(missions)).astype(np.float64)
    err = np.abs(rng.normal(0, err_scale, n))
    mu_true = np.clip(rng.uniform(0.5, 1.0, n), 0, 1)
    return {
        "mission": mission,
        "timestamp": timestamp,
        "mu_pred": np.clip(mu_true + rng.normal(0, 0.02, n), 0, 1),
        "mu_pred_rgbonly": np.clip(mu_true + rng.normal(0, 0.05, n), 0, 1),
        "mu_true_min": mu_true,
        "mu_true_mean": mu_true,
        "mu_err_mean": err,
        "n_visible": rng.integers(0, 40, n),
        "evidential_unc": np.full(n, np.nan),
        "evidential_unc_rgbonly": np.full(n, np.nan),
        "bev_feat": (rng.normal(0, 1, (n, BEV_DIM)) + feat_shift).astype(np.float32),
        "bev_feat_rgbonly": rng.normal(0, 1, (n, BEV_DIM)).astype(np.float32),
        "dino_feat": (rng.normal(0, 1, (n, DINO_DIM)) + feat_shift).astype(np.float32),
    }


@pytest.fixture
def features_dir(tmp_path):
    rng = np.random.default_rng(0)
    tiers = {
        "id_train": make_tier(rng, ["TRA-1", "TRA-2"], n_per=200),
        "id_heldout": make_tier(rng, ["HELD-1", "HELD-2", "HELD-3"]),
        "near_forest": make_tier(rng, ["NF-1", "NF-2"], feat_shift=0.5, err_scale=0.08),
        "far_snow": make_tier(rng, ["FS-1", "FS-2"], feat_shift=3.0, err_scale=0.08),
    }
    for name, data in tiers.items():
        np.savez(tmp_path / f"{name}.npz", **data)
    return str(tmp_path)
