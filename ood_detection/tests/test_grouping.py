import numpy as np
import pytest

from ood_detection.eval.grouping import block_bootstrap, event_ids, time_blocks


def test_time_blocks_splits_mission_into_fixed_windows():
    mission = np.array(["A"] * 90)
    timestamp = np.arange(90, dtype=np.float64)
    ids = time_blocks(mission, timestamp, block_s=30.0)
    assert len(np.unique(ids)) == 3
    assert (np.bincount(ids) == 30).all()

    # blocks are contiguous in time
    for b in np.unique(ids):
        t = timestamp[ids == b]
        assert t.max() - t.min() < 30.0


def test_time_blocks_ids_unique_across_missions():
    mission = np.array(["A"] * 40 + ["B"] * 40)
    timestamp = np.concatenate([np.arange(40), np.arange(40)]).astype(np.float64)
    ids = time_blocks(mission, timestamp, block_s=30.0)
    assert set(ids[mission == "A"]).isdisjoint(ids[mission == "B"])
    assert (ids >= 0).all()


def test_time_blocks_handles_time_gaps():
    mission = np.array(["A"] * 5)
    timestamp = np.array([0.0, 1.0, 2.0, 100.0, 101.0])
    ids = time_blocks(mission, timestamp, block_s=30.0)
    assert len(np.unique(ids)) == 2


def test_event_ids_clusters_by_gap():
    mission = np.array(["A"] * 5)
    timestamp = np.array([0.0, 1.0, 2.0, 50.0, 51.0])
    flag = np.ones(5, dtype=bool)
    ids, n = event_ids(mission, timestamp, flag, gap_s=10.0)
    assert n == 2
    assert list(ids) == [0, 0, 0, 1, 1]


def test_event_ids_splits_on_mission_boundary():
    mission = np.array(["A", "A", "B", "B"])
    timestamp = np.array([0.0, 1.0, 2.0, 3.0])
    ids, n = event_ids(mission, timestamp, np.ones(4, dtype=bool), gap_s=10.0)
    assert n == 2
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[0] != ids[2]


def test_event_ids_unflagged_and_empty():
    mission = np.array(["A"] * 4)
    timestamp = np.arange(4, dtype=np.float64)
    flag = np.array([True, False, False, True])
    ids, n = event_ids(mission, timestamp, flag, gap_s=10.0)
    assert n == 1  # gap between flagged windows is 3s <= 10s: one event
    assert ids[1] == -1 and ids[2] == -1

    ids, n = event_ids(mission, timestamp, np.zeros(4, dtype=bool))
    assert n == 0 and (ids == -1).all()


def test_block_bootstrap_ci_covers_mean_and_is_deterministic():
    rng = np.random.default_rng(1)
    x = rng.normal(5.0, 1.0, 600)
    blocks = np.arange(600) // 10
    ci1 = block_bootstrap(lambda idx: float(x[idx].mean()), blocks, n_boot=300, seed=7)
    ci2 = block_bootstrap(lambda idx: float(x[idx].mean()), blocks, n_boot=300, seed=7)
    assert ci1 == ci2
    assert ci1[0] < 5.0 < ci1[1]
    assert ci1[1] - ci1[0] < 0.5


def test_block_bootstrap_single_block_degenerates():
    x = np.arange(10, dtype=np.float64)
    lo, hi = block_bootstrap(lambda idx: float(x[idx].mean()), np.zeros(10, int), n_boot=20)
    assert lo == hi == pytest.approx(4.5)


def test_block_bootstrap_all_nan_metric():
    lo, hi = block_bootstrap(lambda idx: float("nan"), np.arange(10), n_boot=20)
    assert np.isnan(lo) and np.isnan(hi)


def test_block_bootstrap_clustered_data_widens_ci():
    """Correlated blocks must yield wider CIs than treating windows as iid."""
    rng = np.random.default_rng(2)
    block_effect = np.repeat(rng.normal(0, 1.0, 40), 15)
    x = block_effect + rng.normal(0, 0.1, 600)
    blocks = np.arange(600) // 15
    lo_b, hi_b = block_bootstrap(lambda idx: float(x[idx].mean()), blocks, n_boot=300, seed=0)
    lo_i, hi_i = block_bootstrap(lambda idx: float(x[idx].mean()), np.arange(600), n_boot=300, seed=0)
    assert (hi_b - lo_b) > 2 * (hi_i - lo_i)
