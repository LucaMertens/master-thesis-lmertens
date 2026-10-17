"""W-RIZZ (Schreiber et al. 2024): ordinal pairs from RHE mu labels, hinge ranking loss, HDR."""
import torch
import torch.nn as nn

from .metrics import gather_sum

HEADLINE_TAU = 0.25
# Fixed base seed so HDR is measured on the same pairs every time
PAIR_SEED = 0x217717


def _take(idx, n, generator):
    """First ``n`` candidates, resampled with replacement when the pool is short."""
    if n <= 0 or len(idx) == 0:
        return idx[:0]
    if len(idx) >= n:
        return idx[:n]
    extra = torch.randint(len(idx), (n - len(idx),), device=idx.device, generator=generator)
    return torch.cat([idx, idx[extra]])


def _candidates(mu, valid, xy, n, deadband, tie_band, cross_frac, min_sep, generator):
    """Draw ``n`` candidate pairs and label them; the caller balances ties and inequalities."""
    B, H = mu.shape
    def draw(hi):
        return torch.randint(hi, (n,), device=mu.device, generator=generator)

    b_a, h_a, h_b = draw(B), draw(H), draw(H)
    cross = torch.rand(n, device=mu.device, generator=generator) < cross_frac
    # offset keeps a cross pair out of its own window
    b_b = torch.where(cross, (b_a + 1 + draw(B - 1)) % B, b_a) if B > 1 else b_a

    ok = valid[b_a, h_a] & valid[b_b, h_b]
    sep = (xy[b_a, h_a] - xy[b_b, h_b]).norm(dim=-1)
    ok &= cross | (sep >= min_sep)
    d = mu[b_b, h_b] - mu[b_a, h_a]
    # drop the ambiguous band between tie and inequality
    ok &= (d.abs() > deadband) | (d.abs() <= tie_band)
    b_a, h_a, b_b, h_b, d = (v[ok] for v in (b_a, h_a, b_b, h_b, d))
    return b_a, h_a, b_b, h_b, torch.sign(d) * (d.abs() > deadband)


def sample_pairs(mu, valid, xy, n_pairs, deadband, tie_band, cross_frac, min_sep,
                 ineq_frac, ineq_intra_only=False, generator=None, oversample=64):
    """Draw ordinal pairs of labelled path cells from a batch.

    ``mu``/``valid`` are (B, H), ``xy`` is (B, H, 2). Pairs are intra-window (at least
    ``min_sep`` m apart) or cross-window. Gap above ``deadband`` is an inequality, below
    ``tie_band`` a tie, in between dropped. Returns ((b_a, h_a), (b_b, h_b), t) with
    t in {-1, 0, 1}, t = 1 meaning b is the more traversable cell.

    ``ineq_intra_only`` draws inequalities intra-window only (cross-mission inequalities
    mostly encode recording identity, 93% are solved by the mission mean mu); ties keep
    the cross share.
    """
    n = n_pairs * oversample
    pool = _candidates(mu, valid, xy, n, deadband, tie_band, cross_frac, min_sep, generator)
    # separate all-intra draw so inequalities are not starved
    ineq_pool = (
        _candidates(mu, valid, xy, n, deadband, tie_band, 0.0, min_sep, generator)
        if ineq_intra_only else pool
    )

    # RHE mu is mostly high, so uniform pairs are almost all ties. Hold the
    # inequality ratio exactly (with replacement if a batch is short).
    ineq = _take((ineq_pool[4] != 0).nonzero().squeeze(1), round(ineq_frac * n_pairs), generator)
    eq = _take((pool[4] == 0).nonzero().squeeze(1), n_pairs - len(ineq), generator)
    b_a, h_a, b_b, h_b, t = (
        torch.cat([ineq_pool[i][ineq], pool[i][eq]]) for i in range(5))
    return (b_a, h_a), (b_b, h_b), t


def rizz_terms(p_a, p_b, t, margin):
    """Per-pair squared hinge (W-RIZZ Eq. 2), unreduced so callers can weight pairs."""
    d = p_b - p_a
    return torch.where(t != 0, (margin - t * d).clamp_min(0).square(), d.square())


def rizz_loss(p_a, p_b, t, margin):
    """Mean squared-hinge over a set of pairs."""
    if p_a.numel() == 0:
        return p_a.sum()
    return rizz_terms(p_a, p_b, t, margin).mean()


def pair_loss(pred_mu, target_mu, xy, valid, margin, **pair_kw):
    """Sample ordinal pairs from the labels and score the predictions on them."""
    a, b, t = sample_pairs(target_mu, valid, xy, **pair_kw)
    return rizz_loss(pred_mu[a], pred_mu[b], t, margin)


def hdr_counts(p_a, p_b, t, taus):
    """(n_taus, 3, 2) of (disagreements, pairs) over all / tie / inequality pairs."""
    d = p_b - p_a
    groups = torch.stack([torch.ones_like(t, dtype=torch.bool), t == 0, t != 0])
    out = torch.zeros(len(taus), 3, 2, device=d.device)
    for i, tau in enumerate(taus):
        wrong = (torch.sign(d) * (d.abs() > tau)) != t
        out[i, :, 0] = (groups & wrong).sum(-1)
        out[i, :, 1] = groups.sum(-1)
    return out


def balanced_hdr(counts):
    """Mean of tie and inequality disagreement rates from one threshold's (3, 2) counts.

    0.5 for constant or random predictions; an empty group counts as 0.5."""
    rates = [c[0] / c[1] if c[1] > 0 else torch.full_like(c[0], 0.5)
             for c in (counts[1], counts[2])]
    return 0.5 * (rates[0] + rates[1])


# Evaluation pair spec, fixed so HDR is comparable across runs:
#   ""       cross-window pairs included
#   "_intra" intra-window pairs only (per-mission mu offset cancels)
EVAL_SPECS = {
    "": dict(deadband=0.25, tie_band=0.125, cross_frac=0.5, min_sep=2.0,
             ineq_frac=0.5, ineq_intra_only=False),
    "_intra": dict(deadband=0.25, tie_band=0.125, cross_frac=0.0, min_sep=2.0,
                   ineq_frac=0.5, ineq_intra_only=True),
}


class RizzMetrics(nn.Module):
    """Human Disagreement Rate (W-RIZZ Eq. 6) over pairs drawn from label mu with EVAL_SPECS."""

    def __init__(self, n_pairs, taus=(0.1, 0.25, 0.5)):
        super().__init__()
        self.n_pairs = n_pairs
        self.specs = list(EVAL_SPECS)
        self.register_buffer("taus", torch.tensor(taus), persistent=False)
        shape = (len(EVAL_SPECS), len(taus), 3, 2)
        self.register_buffer("val_counts", torch.zeros(shape), persistent=False)
        self.register_buffer("train_counts", torch.zeros(shape), persistent=False)

    @torch.no_grad()
    def update(self, pl_module, states, trav_map, target_trav, trav_weights,
               batch_idx, train):
        pred, mask = pl_module._sample_path(states, trav_map)
        valid = mask.bool() & (trav_weights[..., 0] > 0)
        if valid.sum() < 2:
            return
        counts = self.train_counts if train else self.val_counts
        for s, name in enumerate(self.specs):
            gen = torch.Generator(device=pred.device)
            gen.manual_seed(PAIR_SEED + batch_idx + 977 * s)
            a, b, t = sample_pairs(target_trav[..., 0], valid, states[..., :2],
                                   n_pairs=self.n_pairs, generator=gen,
                                   **EVAL_SPECS[name])
            if t.numel() == 0:
                continue
            counts[s] += hdr_counts(pred[..., 0][a], pred[..., 0][b], t, self.taus)

    def log(self, pl_module, prefix, train=False):
        all_counts = self.train_counts if train else self.val_counts
        for s, name in enumerate(self.specs):
            self._log_spec(pl_module, f"{prefix}/hdr{name}", all_counts[s])
        all_counts.zero_()

    def _log_spec(self, pl_module, key, counts):
        # clone: on a single device all_gather returns the buffer itself, which is reset later
        gathered = gather_sum(pl_module, counts).clone()
        if gathered[0, 0, 1] == 0:
            return
        for i, tau in enumerate(self.taus.tolist()):
            for j, group in enumerate(("", "_eq", "_ineq")):
                wrong, total = gathered[i, j]
                if total > 0:
                    pl_module.log(f"{key}{group}_{tau:g}", wrong / total, sync_dist=False)
            if abs(tau - HEADLINE_TAU) < 1e-9:
                pl_module.log(key, gathered[i, 0, 0] / gathered[i, 0, 1], sync_dist=False)
                pl_module.log(f"{key}_balanced", balanced_hdr(gathered[i]), sync_dist=False)
        pl_module.log(f"{key}_pair_ineq_frac",
                      gathered[0, 2, 1] / gathered[0, 0, 1], sync_dist=False)
