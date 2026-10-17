import torch

from traversability_network.adaptations import rizz

KW = dict(deadband=0.25, tie_band=0.125, cross_frac=0.5, min_sep=0.5, ineq_frac=0.5)


def make_batch(B=4, H=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    mu = torch.rand(B, H, generator=g)
    xy = torch.rand(B, H, 2, generator=g) * 8
    return mu, torch.ones(B, H, dtype=torch.bool), xy


def test_labels_match_the_mu_gap():
    mu, valid, xy = make_batch()
    a, b, t = rizz.sample_pairs(mu, valid, xy, n_pairs=200, **KW)
    d = mu[b] - mu[a]
    assert torch.equal(t[d > KW["deadband"]], torch.ones_like(t[d > KW["deadband"]]))
    assert torch.equal(t[d < -KW["deadband"]], -torch.ones_like(t[d < -KW["deadband"]]))
    assert (d[t == 0].abs() <= KW["deadband"]).all()


def test_ambiguous_band_is_dropped():
    mu, valid, xy = make_batch()
    a, b, t = rizz.sample_pairs(mu, valid, xy, n_pairs=400, **KW)
    gap = (mu[b] - mu[a]).abs()
    assert not ((gap > KW["tie_band"]) & (gap <= KW["deadband"])).any()


def test_intra_pairs_are_separated_and_cross_pairs_differ():
    mu, valid, xy = make_batch()
    (ba, ha), (bb, hb), _ = rizz.sample_pairs(mu, valid, xy, n_pairs=400, **KW)
    intra = ba == bb
    assert intra.any() and (~intra).any()
    sep = (xy[ba, ha] - xy[bb, hb]).norm(dim=-1)
    assert (sep[intra] >= KW["min_sep"]).all()


def test_gated_cells_never_enter_a_pair():
    mu, valid, xy = make_batch()
    valid[:, ::2] = False
    (ba, ha), (bb, hb), _ = rizz.sample_pairs(mu, valid, xy, n_pairs=300, **KW)
    assert valid[ba, ha].all() and valid[bb, hb].all()


def test_inequality_share_is_enforced_on_skewed_labels():
    # RHE-like labels: almost everything near 1.0, so a uniform draw is nearly all ties
    mu, valid, xy = make_batch()
    mu = torch.where(mu < 0.97, torch.ones_like(mu), mu * 0.2)
    _, _, t = rizz.sample_pairs(mu, valid, xy, n_pairs=400, **KW)
    assert 0.4 < (t != 0).float().mean() < 0.6


def test_ineq_intra_only_keeps_inequalities_in_one_window():
    mu, valid, xy = make_batch()
    (ba, ha), (bb, hb), t = rizz.sample_pairs(
        mu, valid, xy, n_pairs=400, ineq_intra_only=True, **KW)
    assert (ba[t != 0] == bb[t != 0]).all(), "inequality pair spans two windows"
    assert (ba[t == 0] != bb[t == 0]).any(), "ties lost their cross-window share"
    assert 0.4 < (t != 0).float().mean() < 0.6, "ratio not held"


def test_loss_is_zero_once_the_margin_is_met():
    t = torch.tensor([1.0, -1.0, 0.0])
    p_a = torch.tensor([0.2, 0.9, 0.5])
    p_b = torch.tensor([0.8, 0.3, 0.5])
    assert rizz.rizz_loss(p_a, p_b, t, margin=0.5) == 0.0
    # ordering reversed on the first pair
    assert rizz.rizz_loss(p_b, p_a, t, margin=0.5) > 0.0


def test_loss_penalizes_a_constant_prediction():
    mu, valid, xy = make_batch()
    a, b, t = rizz.sample_pairs(mu, valid, xy, n_pairs=500, **KW)
    const = torch.full_like(mu, 0.5)
    collapsed = rizz.rizz_loss(const[a], const[b], t, 0.5)
    perfect = rizz.rizz_loss(mu[a], mu[b], t, 0.5)
    assert collapsed > perfect


def test_hdr_counts_split_by_relation():
    t = torch.tensor([1.0, 0.0, -1.0, 1.0])
    p_a = torch.tensor([0.1, 0.5, 0.9, 0.9])
    p_b = torch.tensor([0.9, 0.5, 0.1, 0.1])  # last pair predicted backwards
    counts = rizz.hdr_counts(p_a, p_b, t, torch.tensor([0.25]))
    assert counts[0, 0].tolist() == [1, 4]  # all pairs
    assert counts[0, 1].tolist() == [0, 1]  # ties
    assert counts[0, 2].tolist() == [1, 3]  # inequalities


def test_pairs_are_reproducible_under_a_seeded_generator():
    mu, valid, xy = make_batch()
    def draw():
        return rizz.sample_pairs(
            mu, valid, xy, n_pairs=100, generator=torch.Generator().manual_seed(7), **KW)
    (a1, h1), _, t1 = draw()
    (a2, h2), _, t2 = draw()
    assert torch.equal(a1, a2) and torch.equal(h1, h2) and torch.equal(t1, t2)
