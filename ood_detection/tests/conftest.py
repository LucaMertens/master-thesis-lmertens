"""Synthetic features dir mimicking ood_detection.features.extract output."""

import numpy as np
import pytest

from ood_detection.eval.run import OOD_TIERS

BEV_DIM, DINO_DIM, N_KEEP = 8, 16, 32
GRID = 12
XBOUND, YBOUND = [-2.0, 4.0, 0.5], [-3.0, 3.0, 0.5]


def feature_space(dim, seed):
    """Anisotropic covariance, nonzero mean and a shift direction off both (else Mahalanobis = Euclidean)."""
    r = np.random.default_rng(seed)
    mixing = r.normal(0, 1, (dim, dim)) @ np.diag(np.linspace(1.0, 0.15, dim))
    mean = np.zeros(dim)
    mean[0] = 8.0
    direction = r.normal(0, 1, dim)
    return mixing, mean, direction / np.linalg.norm(direction)


BEV_SPACE, DINO_SPACE = feature_space(BEV_DIM, 1), feature_space(DINO_DIM, 2)


def make_feat(rng, n, space, shift):
    mixing, mean, direction = space
    x = rng.normal(0, 1, (n, mixing.shape[0])) @ mixing + mean + shift * direction
    return x.astype(np.float32)


def cell_centers():
    """The model's voxel grid: both axes descending, as create_voxels builds them."""
    x, y = np.arange(*XBOUND)[::-1][:GRID], np.arange(*YBOUND)[::-1][:GRID]
    return np.stack(np.meshgrid(x, y, indexing="ij"), -1).astype(np.float32)


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
        "n_cells": np.full(n, N_KEEP),
        "cells": make_feat(rng, n * N_KEEP, BEV_SPACE, feat_shift).reshape(n, N_KEEP, BEV_DIM),
        "cells_rgbonly": make_feat(rng, n * N_KEEP, BEV_SPACE, 0.0).reshape(n, N_KEEP, BEV_DIM),
        "dino_feat": make_feat(rng, n, DINO_SPACE, feat_shift),
        # viewpoint detector: a mu map, what it observes, and where the window was
        "mu_map": rng.uniform(0, 1, (n, GRID, GRID)).astype(np.float16),
        "observed_map": np.ones((n, GRID, GRID), bool),
        "pose": np.stack([np.tile(np.arange(n_per) * 0.04, len(missions)),
                          np.zeros(n), np.zeros(n)], -1),
        "centers": cell_centers(),
        "xbound": np.array(XBOUND),
        "ybound": np.array(YBOUND),
    }


# Synthetic tiers reuse live tier names (eval only scores declared tiers): NEAR mild, FAR 3-sigma
NEAR_TIER = OOD_TIERS[0]
FAR_TIER = next(t for t in OOD_TIERS if t.startswith("far_"))


@pytest.fixture
def features_dir(tmp_path):
    rng = np.random.default_rng(0)
    tiers = {
        "id_train": make_tier(rng, ["TRA-1", "TRA-2"], n_per=200),
        "id_heldout": make_tier(rng, ["HELD-1", "HELD-2", "HELD-3"]),
        NEAR_TIER: make_tier(rng, ["NF-1", "NF-2"], feat_shift=0.5, err_scale=0.08),
        FAR_TIER: make_tier(rng, ["FS-1", "FS-2"], feat_shift=3.0, err_scale=0.08),
    }
    for name, data in tiers.items():
        np.savez(tmp_path / f"{name}.npz", **data)
    return str(tmp_path)
