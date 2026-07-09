"""Unit tests for the low-mu oversampler (GrandTourDataset._flagged_windows /
_oversampled_indices). Builds bare instances via object.__new__ so no dataset on
disk is needed; only numpy is exercised."""

import numpy as np
import pytest

from traversability_network.data.grandtour_dataset import GrandTourDataset


def make_ds(rosbags, *, n_frames=1, horizon=5, dt=0.1,
            mu_max=0.6, obs_min=0.6, fraction=0.4):
    """A GrandTourDataset with only the attributes the sampler touches."""
    ds = object.__new__(GrandTourDataset)
    ds.rosbags = rosbags
    ds.n_frames = n_frames
    ds.horizon = horizon
    ds.dt = dt
    ds.os_mu_max = mu_max
    ds.os_obs_min = obs_min
    ds.OVERSAMPLE_FRACTION = fraction
    ds.grid_bounds = {"xbound": [-2.0, 8.0, 0.1], "ybound": [-5.0, 5.0, 0.1]}
    ds.bag_sizes = [0]
    for bag in rosbags:
        ds.bag_sizes.append(ds.bag_sizes[-1] + len(bag["image_timestamp"]) - n_frames + 1)
    return ds


def make_bag(mu, obs, *, dt=0.1, n_frames=1, position=None):
    """A bag whose images coincide with its states (one window per state).
    All states sit at the origin unless position (N, 3) is given, so the
    spatial in-grid filter passes and the tests exercise the time logic."""
    mu = np.asarray(mu, float)
    st = np.arange(len(mu)) * dt
    n = len(mu)
    return {
        "states_timestamp": st,
        "traversability": np.stack([mu, np.zeros_like(mu)], axis=1),
        "mu_observability": np.asarray(obs, float),
        "image_timestamp": st,
        "position": np.zeros((n, 3)) if position is None else np.asarray(position, float),
        "quaternion": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
    }


def brute_flags(ds):
    """Independent reference for _flagged_windows."""
    n_flag = min(ds.horizon, 300)
    out = []
    for bag in ds.rosbags:
        st = np.asarray(bag["states_timestamp"])
        mu = np.asarray(bag["traversability"])[:, 0]
        observability = np.asarray(bag["mu_observability"])
        nw = len(bag["image_timestamp"]) - ds.n_frames + 1
        for t in np.asarray(bag["image_timestamp"])[:nw]:
            in_win = (st >= t) & (st < t + n_flag * ds.dt)
            qual = (mu <= ds.os_mu_max) & (observability >= ds.os_obs_min)
            out.append(bool((in_win & qual).any()))
    return np.array(out)


def test_flagged_matches_bruteforce():
    mu = np.ones(20)
    mu[10:15] = 0.3  # low-mu states at t in [1.0, 1.5)
    ds = make_ds([make_bag(mu, np.ones(20))], horizon=5)
    flags = ds._flagged_windows()
    assert np.array_equal(flags, brute_flags(ds))
    # window at t=1.0 sees the low-mu states; window at t=0.0 (covers [0,0.5)) does not
    assert flags[10] and not flags[0]


def test_observability_gate_excludes_unobserved_low_mu():
    mu = np.full(10, 0.2)  # all low mu
    observability = np.full(10, 0.1)  # but unobservable -> must not flag
    ds = make_ds([make_bag(mu, observability)], horizon=5)
    assert not ds._flagged_windows().any()


def test_mu_threshold_boundary():
    # mu exactly at the max qualifies (<=); just above does not
    ds_eq = make_ds([make_bag(np.full(3, 0.6), np.ones(3))], horizon=3)
    ds_gt = make_ds([make_bag(np.full(3, 0.61), np.ones(3))], horizon=3)
    assert ds_eq._flagged_windows().all()
    assert not ds_gt._flagged_windows().any()


def test_out_of_grid_low_mu_not_flagged():
    # low-mu state 50m away: temporally inside the horizon of earlier windows
    # but outside their BEV grid, so only the window anchored on it flags
    mu = np.ones(20)
    mu[10] = 0.3
    position = np.zeros((20, 3))
    position[10, 0] = 50.0
    ds = make_ds([make_bag(mu, np.ones(20), position=position)], horizon=5)
    flags = ds._flagged_windows()
    assert flags[10]
    assert not flags[6:10].any()


def test_flags_concatenate_across_bags():
    low = make_bag(np.full(4, 0.3), np.ones(4))
    high = make_bag(np.ones(4), np.ones(4))
    ds = make_ds([low, high], horizon=2)
    flags = ds._flagged_windows()
    assert len(flags) == ds.bag_sizes[-1]
    assert flags[:4].any() and not flags[4:].any()


def _frac_flagged(ds, indices):
    flagged_set = set(np.flatnonzero(ds._flagged_windows()).tolist())
    n = sum(i in flagged_set for i in indices)
    return n / len(indices)


def test_oversampled_reaches_target_fraction():
    flags = np.zeros(100, bool)
    flags[:10] = True  # 10% natural
    ds = make_ds([], fraction=0.4)
    ds._flagged_windows = lambda: flags
    idx = ds._oversampled_indices()
    # kept (90) + n_target where n_target/(90+n_target)=0.4 -> n_target=60
    assert len(idx) == 150
    assert _frac_flagged(ds, idx) == pytest.approx(0.4, abs=0.01)
    # every non-flagged window appears exactly once
    kept = np.flatnonzero(~flags)
    counts = np.bincount(idx, minlength=100)
    assert np.all(counts[kept] == 1)


def test_no_oversampling_when_natural_rate_exceeds_target():
    flags = np.zeros(100, bool)
    flags[:50] = True  # 50% natural already above the 40% target
    ds = make_ds([], fraction=0.4)
    ds._flagged_windows = lambda: flags
    idx = ds._oversampled_indices()
    assert idx == list(range(100))  # identity, never downsamples


def test_no_flagged_windows_is_identity():
    flags = np.zeros(30, bool)
    ds = make_ds([], fraction=0.4)
    ds._flagged_windows = lambda: flags
    assert ds._oversampled_indices() == list(range(30))


def test_oversampling_is_deterministic():
    flags = np.zeros(80, bool)
    flags[:8] = True
    ds = make_ds([], fraction=0.5)
    ds._flagged_windows = lambda: flags
    assert ds._oversampled_indices() == ds._oversampled_indices()


def test_higher_fraction_replicates_more():
    flags = np.zeros(100, bool)
    flags[:10] = True
    ds = make_ds([], fraction=0.4)
    ds._flagged_windows = lambda: flags
    low = len(ds._oversampled_indices())
    ds.OVERSAMPLE_FRACTION = 0.6
    high = len(ds._oversampled_indices())
    assert high > low
    assert _frac_flagged(ds, ds._oversampled_indices()) == pytest.approx(0.6, abs=0.01)


def test_replication_cycles_all_flagged_windows():
    # resize should cycle through every flagged index, not just the first
    flags = np.zeros(50, bool)
    flags[[3, 7, 41]] = True
    ds = make_ds([], fraction=0.7)
    ds._flagged_windows = lambda: flags
    counts = np.bincount(ds._oversampled_indices(), minlength=50)
    assert np.all(counts[[3, 7, 41]] >= 1)
