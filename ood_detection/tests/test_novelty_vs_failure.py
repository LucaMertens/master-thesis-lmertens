import os

import numpy as np
import pandas as pd
import pytest

from ood_detection.eval.novelty_vs_failure import (
    FAIL_ERR,
    analysis_row,
    binary_auroc,
    main,
    pooled_windows,
    quadrant_table,
    ranking_comparison,
)


def make_pooled(rng, n=2000):
    """Pooled window set with independent novelty and failure labels."""
    novel = rng.random(n) < 0.5
    wrong = rng.random(n) < 0.1
    err = np.where(wrong, FAIL_ERR + rng.uniform(0, 0.2, n), rng.uniform(0, FAIL_ERR * 0.9, n))
    return {
        "mission": np.where(novel, "OOD-1", "ID-1"),
        "timestamp": np.arange(n, dtype=np.float64),
        "mu_err_mean": err,
        "mu_true_min": rng.uniform(0.65, 1.0, n),
        "n_visible": np.full(n, 20),
        "novel": novel,
    }, novel, wrong


def test_binary_auroc_edge_cases():
    s = np.arange(10, dtype=np.float64)
    assert np.isnan(binary_auroc(s, np.ones(10, bool)))
    assert np.isnan(binary_auroc(s, np.zeros(10, bool)))
    assert binary_auroc(s, s >= 5) == 1.0


def test_novelty_detector_scores_high_on_novelty_only():
    rng = np.random.default_rng(0)
    pooled, novel, wrong = make_pooled(rng)
    scores = novel + rng.normal(0, 0.1, len(novel))
    row, _ = analysis_row(scores, pooled, n_boot=50)
    assert row["auroc_novelty"] > 0.95
    assert abs(row["auroc_wrong"] - 0.5) < 0.05
    assert row["auroc_novelty_lo"] <= row["auroc_novelty"] <= row["auroc_novelty_hi"]


def test_failure_detector_scores_high_on_failure_only():
    rng = np.random.default_rng(1)
    pooled, novel, wrong = make_pooled(rng)
    scores = wrong + rng.normal(0, 0.1, len(wrong))
    row, _ = analysis_row(scores, pooled, n_boot=50)
    assert row["auroc_wrong"] > 0.95
    assert row["auroc_wrong_id"] > 0.95
    assert abs(row["auroc_novelty"] - 0.5) < 0.05
    assert row["spearman_err"] > 0.25  # bounded by the ~10% positive rate


def test_ranking_inversion_detected():
    rng = np.random.default_rng(2)
    pooled, novel, wrong = make_pooled(rng)
    rows = []
    for name, scores in [
        ("novelty_det", novel + rng.normal(0, 0.1, len(novel))),
        ("failure_det", wrong + rng.normal(0, 0.1, len(wrong))),
    ]:
        row, _ = analysis_row(scores, pooled, n_boot=10)
        rows.append({"detector": name, **row})
    ranking, tau = ranking_comparison(pd.DataFrame(rows))
    assert tau == -1.0
    assert ranking.iloc[0]["detector"] == "novelty_det"
    assert ranking.iloc[0]["rank_wrong"] == 2


def test_hazard_auroc_and_event_counts():
    rng = np.random.default_rng(3)
    pooled, novel, wrong = make_pooled(rng)
    # one contiguous hazard event of 20 windows in the ID mission
    idx = np.flatnonzero(~novel)[:20]
    pooled["mu_true_min"][idx] = 0.4
    scores = np.zeros(len(novel))
    scores[idx] = 1.0
    row, _ = analysis_row(scores, pooled, n_boot=10)
    assert row["auroc_hazard"] > 0.99
    assert row["n_hazard"] == 20
    assert row["n_hazard_events"] <= 3  # contiguous timestamps, few events


def test_gating_drops_invisible_and_nonfinite_windows():
    rng = np.random.default_rng(4)
    pooled, novel, wrong = make_pooled(rng, n=1000)
    pooled["n_visible"][:100] = 0
    pooled["mu_err_mean"][100:150] = np.nan
    scores = rng.normal(0, 1, 1000)
    scores[150:170] = np.nan
    row, (s, *_rest) = analysis_row(scores, pooled, n_boot=10)
    assert row["n"] == 1000 - 100 - 50 - 20
    assert np.isfinite(s).all()


def test_quadrant_table_hand_case():
    # ID: 100 right windows scoring 0..99 -> q95 threshold is 95.05ish
    s = np.concatenate([np.arange(100.0), np.array([200.0, 0.0, 200.0, 0.0])])
    novel = np.array([False] * 100 + [False, False, True, True])
    wrong = np.array([False] * 100 + [True, True, False, False])
    rows = {r["quadrant"]: r for r in quadrant_table("d", s, novel, wrong)}
    assert rows["id_right"]["n"] == 100
    assert rows["id_wrong"]["alarm_rate"] == 0.5   # one of two id_wrong above thresh
    assert rows["ood_right"]["alarm_rate"] == 0.5
    assert rows["id_right"]["alarm_rate"] < 0.1
    assert sum(r["n"] for r in rows.values()) == 104


def test_pooled_windows_assembles_novel_label(features_dir):
    sets, pooled = pooled_windows(features_dir)
    names = [t for t, _ in sets]
    assert names[0] == "id_heldout" and "far_snow" in names
    n_id = len(np.load(os.path.join(features_dir, "id_heldout.npz"))["timestamp"])
    assert (~pooled["novel"]).sum() == n_id
    assert pooled["novel"].sum() == len(pooled["mission"]) - n_id


def test_main_end_to_end(features_dir):
    df, quadrants = main(["--features-dir", features_dir, "--n-boot", "20"])
    out = os.path.join(features_dir, "eval")
    assert os.path.exists(os.path.join(out, "novelty_vs_failure.csv"))
    assert os.path.exists(os.path.join(out, "novelty_vs_failure_quadrants.csv"))
    assert os.path.exists(os.path.join(out, "novelty_vs_failure.png"))

    # raw + ema rows for the three non-evidential detectors
    assert len(df) == 6
    assert set(quadrants["detector"]) == set(df["detector"])
    aurocs = df[["auroc_novelty", "auroc_wrong", "auroc_hazard"]].to_numpy()
    finite = aurocs[np.isfinite(aurocs)]
    assert ((finite >= 0) & (finite <= 1)).all()

    # far_snow features are shifted by 3 sigma: mahalanobis must see novelty
    novelty = df.set_index("detector")["auroc_novelty"]
    assert novelty["mahalanobis_wayfaster"] > 0.7

    # errors only correlate with features via the tiers' higher err_scale, so
    # failure skill must stay below novelty skill
    assert df.set_index("detector")["auroc_wrong"]["mahalanobis_wayfaster"] < novelty["mahalanobis_wayfaster"]

    for _, r in quadrants.groupby("detector"):
        assert r["n"].sum() == df["n"].iloc[0]
        assert r["frac_windows"].sum() == pytest.approx(1.0)
