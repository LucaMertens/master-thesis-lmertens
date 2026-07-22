import os

import numpy as np
import pandas as pd
import pytest

from ood_detection.eval.fingerprint import (
    beats_null,
    centered,
    level_auroc,
    main,
    pairwise_null_aurocs,
    per_mission_aurocs,
    ranking_changes,
    tier_row,
)


def sessions(rng, n_missions, n_per, offset_scale=1.0, extra_shift=0.0, noise=0.1, prefix="M"):
    """Per-mission level offsets (session fingerprint) + within-mission noise."""
    offsets = rng.normal(0, offset_scale, n_missions) + extra_shift
    scores = np.concatenate([o + rng.normal(0, noise, n_per) for o in offsets])
    mission = np.repeat([f"{prefix}{i}" for i in range(n_missions)], n_per)
    return scores, mission


def test_pairwise_null_identical_distributions_near_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, 3000)
    mission = np.repeat(["A", "B", "C"], 1000)
    null = pairwise_null_aurocs(scores, mission)
    assert len(null) == 3
    assert (null >= 0.5).all() and (null < 0.55).all()


def test_pairwise_null_detects_session_offsets():
    rng = np.random.default_rng(1)
    scores, mission = sessions(rng, 4, 200, offset_scale=2.0, noise=0.1)
    null = pairwise_null_aurocs(scores, mission)
    assert len(null) == 6
    assert null.mean() > 0.9


def test_beats_null_hand_cases():
    assert beats_null([0.9], [0.5, 0.7]) == 1.0
    assert beats_null([0.6], [0.5, 0.7]) == 0.5
    assert beats_null([0.5, 0.7], [0.5, 0.7]) == 0.5  # symmetric with ties
    assert np.isnan(beats_null([0.9], []))


def test_centered_removes_mission_level():
    rng = np.random.default_rng(2)
    scores, mission = sessions(rng, 3, 100, offset_scale=5.0)
    c = centered(scores, mission)
    for m in np.unique(mission):
        assert np.median(c[mission == m]) == pytest.approx(0.0, abs=1e-12)
    # within-mission ordering preserved
    idx = mission == "M0"
    assert (np.argsort(c[idx]) == np.argsort(scores[idx])).all()


def test_level_auroc_hand_case():
    id_scores = np.array([0.0] * 5 + [1.0] * 5)
    id_mission = np.array(["A"] * 5 + ["B"] * 5)
    ood_scores = np.array([2.0] * 5 + [3.0] * 5)
    ood_mission = np.array(["X"] * 5 + ["Y"] * 5)
    assert level_auroc(id_scores, id_mission, ood_scores, ood_mission) == 1.0


def test_session_fingerprint_only_shift_is_not_credited():
    """OOD missions whose level offsets come from the same session-noise
    distribution as ID missions must not beat the null. The null is folded
    (>= 0.5) while tier AUROCs are directed, so this lands at or below 0.5."""
    rng = np.random.default_rng(3)
    id_scores, id_mission = sessions(rng, 8, 300, offset_scale=0.3, noise=1.0, prefix="ID")
    ood_scores, ood_mission = sessions(rng, 8, 300, offset_scale=0.3, noise=1.0, prefix="OOD")
    null = pairwise_null_aurocs(id_scores, id_mission)
    row = tier_row(id_scores, id_mission, ood_scores, ood_mission, null)
    assert row["beats_null"] < 0.6
    assert abs(row["excess"]) < 0.15
    assert abs(row["auroc_centered"] - 0.5) < 0.05  # no within-recording signal


def test_genuine_domain_shift_is_credited():
    rng = np.random.default_rng(4)
    id_scores, id_mission = sessions(rng, 8, 300, offset_scale=0.3, noise=1.0, prefix="ID")
    ood_scores, ood_mission = sessions(
        rng, 8, 300, offset_scale=0.3, noise=1.0, extra_shift=3.0, prefix="OOD"
    )
    null = pairwise_null_aurocs(id_scores, id_mission)
    row = tier_row(id_scores, id_mission, ood_scores, ood_mission, null)
    assert row["beats_null"] == 1.0
    assert row["excess"] > 0.2
    assert row["auroc_mission_min"] > row["null_max"]
    assert row["auroc_level"] > 0.9


def test_centering_kills_pure_level_shift_but_keeps_shape_signal():
    rng = np.random.default_rng(5)
    id_scores, id_mission = sessions(rng, 4, 500, offset_scale=0.3, noise=1.0, prefix="ID")
    null = pairwise_null_aurocs(id_scores, id_mission)

    # pure level shift: raw AUROC saturates, centered falls to chance
    level_scores, level_mission = sessions(
        rng, 4, 500, offset_scale=0.3, noise=1.0, extra_shift=5.0, prefix="OOD"
    )
    level = tier_row(id_scores, id_mission, level_scores, level_mission, null)
    assert level["auroc_pooled"] > 0.99
    assert abs(level["auroc_centered"] - 0.5) < 0.05

    # within-recording signal: 30% of each OOD mission's windows shifted high
    shape_scores, shape_mission = sessions(
        rng, 4, 500, offset_scale=0.3, noise=1.0, prefix="OOD"
    )
    for m in np.unique(shape_mission):
        idx = np.flatnonzero(shape_mission == m)[:150]
        shape_scores[idx] += 6.0
    shape = tier_row(id_scores, id_mission, shape_scores, shape_mission, null)
    assert shape["auroc_centered"] > level["auroc_centered"] + 0.02
    assert shape["auroc_centered"] > 0.52


def test_per_mission_aurocs_keys_and_range():
    rng = np.random.default_rng(6)
    id_scores = rng.normal(0, 1, 500)
    ood_scores, ood_mission = sessions(rng, 3, 100, extra_shift=5.0, prefix="OOD")
    res = per_mission_aurocs(id_scores, ood_scores, ood_mission)
    assert set(res) == {"OOD0", "OOD1", "OOD2"}
    assert all(v > 0.99 for v in res.values())


def test_ranking_changes_flags_reordering():
    df = pd.DataFrame([
        {"detector": "a", "tier": "t1", "auroc_pooled": 0.99, "beats_null": 0.5, "excess": 0.0},
        {"detector": "b", "tier": "t1", "auroc_pooled": 0.80, "beats_null": 1.0, "excess": 0.3},
        {"detector": "a", "tier": "t2", "auroc_pooled": 0.9, "beats_null": 0.9, "excess": 0.2},
        {"detector": "b", "tier": "t2", "auroc_pooled": 0.7, "beats_null": 0.6, "excess": 0.1},
        {"detector": "a", "tier": "t3", "auroc_pooled": 0.99, "beats_null": 1.0, "excess": 0.2},
        {"detector": "b", "tier": "t3", "auroc_pooled": 0.90, "beats_null": 1.0, "excess": 0.4},
    ])
    ranks = ranking_changes(df).set_index("tier")
    assert bool(ranks.loc["t1", "changed"])
    assert ranks.loc["t1", "rank_beats_null"] == "b > a"
    assert not bool(ranks.loc["t2", "changed"])
    # saturated beats_null falls back to excess as tiebreak
    assert bool(ranks.loc["t3", "changed"])
    assert ranks.loc["t3", "rank_beats_null"] == "b > a"


def test_main_end_to_end(features_dir):
    df, ranks = main(["--features-dir", features_dir])
    out = os.path.join(features_dir, "eval")
    assert os.path.exists(os.path.join(out, "fingerprint_corrected.csv"))
    assert os.path.exists(os.path.join(out, "fingerprint_per_mission.csv"))
    assert os.path.exists(os.path.join(out, "fingerprint_corrected.png"))

    assert set(df["tier"]) == {"near_forest", "far_snow"}
    assert len(df) == 6  # 3 detectors x 2 tiers
    # far_snow features shifted 3 sigma: mahalanobis detectors must beat the null
    snow = df[(df["tier"] == "far_snow") & (df["detector"] == "mahalanobis_wayfaster")].iloc[0]
    assert snow["beats_null"] > 0.9
    assert snow["excess"] > 0.1
    # per-mission csv covers every (detector, tier, mission) combo
    per_mission = pd.read_csv(os.path.join(out, "fingerprint_per_mission.csv"))
    assert len(per_mission) == 6 * 2
    assert per_mission["auroc"].between(0, 1).all()
    assert set(ranks["tier"]) == {"near_forest", "far_snow"}
