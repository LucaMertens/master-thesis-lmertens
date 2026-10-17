"""In-training histogram metrics must match the offline per-cell ones."""

import numpy as np
import torch
from traversability_network.adaptations import mission_metrics as mm
from traversability_network.adaptations.metrics import MuBinMetrics

N_BINS, N_SCORE = 10, 256


def _sample(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.random(n)
    pred = np.clip(0.45 + 0.3 * true + 0.07 * rng.standard_normal(n), 0, 1)
    return pred, true


def _histograms(pred, true):
    tb = np.clip((true * N_BINS).astype(int), 0, N_BINS - 1)
    pb = np.clip((pred * N_BINS).astype(int), 0, N_BINS - 1)
    counts = np.zeros((N_BINS, N_BINS))
    np.add.at(counts, (tb, pb), 1)
    pred_sum = np.zeros(N_BINS)
    np.add.at(pred_sum, tb, pred)
    scores = np.zeros((2, N_SCORE))
    sb = np.clip((pred * N_SCORE).astype(int), 0, N_SCORE - 1)
    risky = true < mm.RISKY_THRESHOLD
    np.add.at(scores[0], sb[risky], 1)
    np.add.at(scores[1], sb[~risky], 1)
    return counts, pred_sum, scores


def test_hist_metrics_match_exact():
    pred, true = _sample()
    exact = mm.score(pred, true)
    hist = mm.score_hist(*_histograms(pred, true))
    for key in ("calib_slope", "recall_pm1", "false_alarm", "n_cells", "n_risky"):
        assert hist[key] == exact[key], key
    # AUROC ties inside a score bin, so only bin-width agreement
    assert abs(hist["auroc_risky"] - exact["auroc_risky"]) < 1e-3


def test_accumulator_matches_per_mission_scoring():
    pred, true = _sample(seed=1)
    rng = np.random.default_rng(2)
    mission = rng.integers(0, 3, len(true))
    names = ["A", "B", "C"]
    metrics = MuBinMetrics(mission_names=names)
    metrics._accumulate_missions(
        torch.tensor(pred), torch.tensor(true), torch.tensor(mission)
    )
    per, _ = mm.by_mission(pred, true, np.array([names[m] for m in mission]))
    for i, name in enumerate(names):
        got = mm.score_hist(
            metrics.mission_confusion[i].numpy(),
            metrics.mission_pred_sum[i].numpy(),
            metrics.mission_scores[i].numpy(),
        )
        assert got["n_cells"] == per[name]["n_cells"]
        assert abs(got["calib_slope"] - per[name]["calib_slope"]) < 1e-6
        assert abs(got["auroc_risky"] - per[name]["auroc_risky"]) < 1e-3
        assert abs(got["recall_pm1"] - per[name]["recall_pm1"]) < 1e-9
        assert abs(got["false_alarm"] - per[name]["false_alarm"]) < 1e-9


def test_average_excludes_missions_without_risky_cells():
    per = {
        "rich": {"calib_slope": 0.4, "auroc_risky": 0.9, "recall_pm1": 0.5,
                 "false_alarm": 0.02, "n_cells": 10000.0, "n_risky": 5000.0},
        "poor": {"calib_slope": 0.0, "auroc_risky": np.nan, "recall_pm1": 0.99,
                 "false_alarm": 0.10, "n_cells": 10000.0, "n_risky": 3.0},
    }
    mean = mm.average(per)
    assert mean["n_missions_scored"] == 1
    assert mean["calib_slope"] == 0.4
    # false alarm is the one metric every mission can answer
    assert abs(mean["false_alarm"] - 0.06) < 1e-9
