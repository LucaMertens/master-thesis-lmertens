"""Human W-RIZZ pair supervision: loss, side joining, batch sampler."""

import numpy as np
import pytest
import torch

from traversability_network.adaptations import human_pairs as HP

GRID = {"xbound": [-2.0, 8.0, 0.1], "ybound": [-5.0, 5.0, 0.1]}


def make_batch(points, pairs, sides, weights, ts, n_pts=1, m=None):
    """One human block; ``points`` is (B, M, 2) BEV positions."""
    b, mm = np.shape(points)[:2]
    m = m or mm
    xy = torch.zeros(b, m, HP.K_PTS, 2)
    mask = torch.zeros(b, m, HP.K_PTS)
    xy[:, :mm, 0] = torch.tensor(points, dtype=torch.float32)
    mask[:, :mm, 0] = 1.0
    def pad(v, fill, dt):
        return torch.cat([torch.tensor(v, dtype=dt), torch.full((b, m - mm), fill, dtype=dt)], 1)
    return {"xy": xy, "mask": mask, "pair": pad(pairs, -1, torch.int64),
            "side": pad(sides, 0, torch.int64), "weight": pad(weights, 0.0, torch.float32),
            "t": pad(ts, 0.0, torch.float32)}


def const_map(values):
    """A BEV map whose value at every cell of sample i is values[i]."""
    t = torch.zeros(len(values), 2, 100, 100)
    for i, v in enumerate(values):
        t[i, 0] = v
    return t


def test_intra_pair_hinge_is_zero_when_ordered_far_enough():
    # constant maps cannot separate two points in one sample, so use two samples
    human = make_batch([[[0.0, 0.0]], [[0.0, 0.0]]], [[0], [0]], [[0], [1]],
                       [[1.0], [1.0]], [[1.0], [1.0]])
    loss, n, half = HP.human_loss(const_map([0.1, 0.9]), human, GRID, 0.5, 4)
    assert (n, half) == (1, 0)
    assert loss.item() == pytest.approx(0.0)


def test_inequality_in_the_wrong_order_is_penalised():
    human = make_batch([[[0.0, 0.0]], [[0.0, 0.0]]], [[0], [0]], [[0], [1]],
                       [[1.0], [1.0]], [[1.0], [1.0]])
    loss, _, _ = HP.human_loss(const_map([0.9, 0.1]), human, GRID, 0.5, 4)
    assert loss.item() == pytest.approx((0.5 + 0.8) ** 2)


def test_tie_is_squared_difference():
    human = make_batch([[[0.0, 0.0]], [[0.0, 0.0]]], [[0], [0]], [[0], [1]],
                       [[1.0], [1.0]], [[0.0], [0.0]])
    loss, _, _ = HP.human_loss(const_map([0.2, 0.5]), human, GRID, 0.5, 4)
    assert loss.item() == pytest.approx(0.09)


def test_weights_set_the_relative_pull():
    human = make_batch([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]],
                       [[0, 1], [0, 1]], [[0, 0], [1, 1]],
                       [[0.2, 0.8], [0.2, 0.8]], [[0.0, 0.0], [0.0, 0.0]])
    loss, n, _ = HP.human_loss(const_map([0.2, 0.5]), human, GRID, 0.5, 4)
    assert n == 2
    assert loss.item() == pytest.approx(0.09)  # same per-pair value, weights normalise


def test_endpoint_without_its_partner_is_dropped_and_counted():
    human = make_batch([[[0.0, 0.0]], [[0.0, 0.0]]], [[0], [1]], [[0], [0]],
                       [[1.0], [1.0]], [[1.0], [1.0]])
    loss, n, half = HP.human_loss(const_map([0.1, 0.9]), human, GRID, 0.5, 4)
    assert (n, half) == (0, 2)
    assert loss.item() == 0.0


def test_points_outside_the_grid_are_masked_out():
    human = make_batch([[[50.0, 0.0]], [[0.0, 0.0]]], [[0], [0]], [[0], [1]],
                       [[1.0], [1.0]], [[1.0], [1.0]])
    _, n, half = HP.human_loss(const_map([0.1, 0.9]), human, GRID, 0.5, 4)
    assert (n, half) == (0, 1)


def test_region_points_are_averaged():
    human = make_batch([[[0.0, 0.0]], [[0.0, 0.0]]], [[0], [0]], [[0], [1]],
                       [[1.0], [1.0]], [[0.0], [0.0]])
    # second region point on sample 0, on a map that varies in x
    human["xy"][0, 0, 1] = torch.tensor([6.0, 0.0])
    human["mask"][0, 0, 1] = 1.0
    trav = torch.zeros(2, 2, 100, 100)
    trav[0, 0, :50] = 1.0
    loss, _, _ = HP.human_loss(trav, human, GRID, 0.5, 4)
    assert 0.0 < loss.item() < 1.0


class FakeDataset:
    def __init__(self, n, bag_sizes):
        self.indices = list(range(n))
        self.bag_sizes = bag_sizes
        self.rosbags = [None] * (len(bag_sizes) - 1)


def test_batch_sampler_pins_cross_windows_and_covers_the_epoch():
    ds = FakeDataset(64, [0, 64])
    groups = [((0, 3), (0, 40)), ((0, 7), (0, 55))]
    s = HP.HumanBatchSampler(ds, groups, batch_size=8, seed=1)
    assert s.n_pinned == 2
    batches = list(s)
    assert all(len(b) == 8 for b in batches)
    seen = [i for b in batches for i in b]
    assert sorted(seen) == list(range(64))
    for a, b in groups:
        pa, pb = a[1], b[1]
        assert any(pa in batch and pb in batch for batch in batches)


def test_batch_sampler_reshuffles_per_epoch():
    ds = FakeDataset(64, [0, 64])
    s = HP.HumanBatchSampler(ds, [], batch_size=8, seed=1)
    first = list(s)
    s.set_epoch(1)
    assert list(s) != first


def region_batch(regions, pairs, sides, ts):
    """One block with several region points per (sample, slot); ``regions`` is (B, M, K, 2)."""
    b, m, k, _ = np.shape(regions)
    xy = torch.tensor(regions, dtype=torch.float32)
    mask = torch.ones(b, m, k)
    return {"xy": xy, "mask": mask,
            "pair": torch.tensor(pairs, dtype=torch.int64),
            "side": torch.tensor(sides, dtype=torch.int64),
            "weight": torch.ones(b, m), "t": torch.tensor(ts, dtype=torch.float32)}


def ramp_map(b):
    """Score rises linearly with BEV y: 0.0 at y=-5, 1.0 at y=+5."""
    return torch.linspace(1.0, 0.0, 100).expand(b, 2, 100, 100).clone()


def test_point_pairs_match_the_region_mean_for_single_point_regions():
    human = make_batch([[[0.0, 0.0]], [[0.0, 0.0]]], [[0], [0]], [[0], [1]],
                       [[1.0], [1.0]], [[1.0], [1.0]])
    args = (human, GRID, 0.5, 4)
    assert HP.human_loss(const_map([0.9, 0.1]), *args, True)[0].item() == pytest.approx(
        HP.human_loss(const_map([0.9, 0.1]), *args)[0].item())


def test_point_pairs_see_the_spread_the_region_mean_hides():
    # a: y = 0.0 and -4.0 -> 0.5, 0.1 (mean 0.3); b: y = +4.0 and 0.0 -> 0.9, 0.5 (mean 0.7)
    human = region_batch([[[[0.0, 0.0], [0.0, -4.0]]], [[[0.0, 4.0], [0.0, 0.0]]]],
                         [[0], [0]], [[0], [1]], [[1.0], [1.0]])
    trav = ramp_map(2)
    mean_loss, _, _ = HP.human_loss(trav, human, GRID, 0.5, 4)
    pts_loss, _, _ = HP.human_loss(trav, human, GRID, 0.5, 4, True)
    assert mean_loss.item() == pytest.approx(0.01, abs=1e-3)
    assert pts_loss.item() == pytest.approx((0.01 + 0.25 + 0.0 + 0.01) / 4, abs=1e-3)


def test_point_mass_only_matters_between_pairs_of_different_region_size():
    # one pair: the weight scaling cancels in the normalised mean
    human = region_batch([[[[0.0, 0.0], [0.0, -4.0]]], [[[0.0, 4.0], [0.0, 0.0]]]],
                         [[0], [0]], [[0], [1]], [[1.0], [1.0]])
    trav = ramp_map(2)
    one = [HP.human_loss(trav, human, GRID, 0.5, 4, True, m)[0].item()
           for m in ("annotation", "combination")]
    assert one[0] == pytest.approx(one[1])
    # two pairs, one of which kept a single point: now the weighting bites
    two_pairs = region_batch(
        [[[[0.0, 0.0], [0.0, -4.0]], [[0.0, 0.0], [0.0, -4.0]]],
         [[[0.0, 4.0], [0.0, 0.0]], [[0.0, 4.0], [0.0, 0.0]]]],
        [[0, 1], [0, 1]], [[0, 0], [1, 1]], [[1.0, 1.0], [1.0, 1.0]])
    two_pairs["mask"][:, 1, 1] = 0.0  # pair 1 keeps one point a side: 1 combo, not 4
    two = [HP.human_loss(trav, two_pairs, GRID, 0.5, 4, True, m)[0].item()
           for m in ("annotation", "combination")]
    assert two[0] != pytest.approx(two[1])


def test_sampled_point_pairs_converge_on_the_full_combination_mean():
    human = region_batch([[[[0.0, 0.0], [0.0, -4.0]]], [[[0.0, 4.0], [0.0, 0.0]]]],
                         [[0], [0]], [[0], [1]], [[1.0], [1.0]])
    trav = ramp_map(2)
    full = HP.human_loss(trav, human, GRID, 0.5, 4, True)[0].item()
    torch.manual_seed(0)
    drawn = np.mean([HP.human_loss(trav, human, GRID, 0.5, 4, True, "annotation", 256)[0].item()
                     for _ in range(20)])
    assert drawn == pytest.approx(full, abs=0.01)


def test_sampling_gives_every_annotation_the_same_number_of_terms():
    # fixed draw weighs both pairs equally, so "combination" mass stops mattering
    human = region_batch(
        [[[[0.0, 0.0], [0.0, -4.0]], [[0.0, 0.0], [0.0, -4.0]]],
         [[[0.0, 4.0], [0.0, 0.0]], [[0.0, 4.0], [0.0, 0.0]]]],
        [[0, 1], [0, 1]], [[0, 0], [1, 1]], [[1.0, 1.0], [1.0, 1.0]])
    human["mask"][:, 1, 1] = 0.0
    torch.manual_seed(0)
    a = HP.human_loss(trav_map := ramp_map(2), human, GRID, 0.5, 4, True, "annotation", 32)[0]
    torch.manual_seed(0)
    b = HP.human_loss(trav_map, human, GRID, 0.5, 4, True, "combination", 32)[0]
    assert a.item() == pytest.approx(b.item())
