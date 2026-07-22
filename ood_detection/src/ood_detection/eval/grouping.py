"""Temporal grouping and cluster bootstrap utilities for correlated windows."""

import numpy as np


def time_blocks(mission, timestamp, block_s=30.0):
    """Fixed-length contiguous time blocks per mission, ids unique across missions."""
    ids = np.full(len(mission), -1, dtype=np.int64)
    offset = 0
    for m in np.unique(mission):
        idx = np.flatnonzero(mission == m)
        raw = np.floor((timestamp[idx] - timestamp[idx].min()) / block_s).astype(np.int64)
        _, dense = np.unique(raw, return_inverse=True)
        ids[idx] = dense + offset
        offset += dense.max() + 1
    return ids


def event_ids(mission, timestamp, flag, gap_s=10.0):
    """Cluster flagged windows into physical events (same mission, gap <= gap_s).
    Returns per-window event ids (-1 where unflagged) and the event count."""
    ids = np.full(len(flag), -1, dtype=np.int64)
    n = 0
    for m in np.unique(mission):
        idx = np.flatnonzero((mission == m) & flag)
        if len(idx) == 0:
            continue
        order = idx[np.argsort(timestamp[idx])]
        new = np.concatenate([[True], np.diff(timestamp[order]) > gap_s])
        ids[order] = np.cumsum(new) - 1 + n
        n += int(new.sum())
    return ids, n


def block_bootstrap(metric_fn, blocks, n_boot=1000, seed=0):
    """95% percentile CI from resampling whole blocks with replacement.
    metric_fn maps an index array (into the original windows) to a float."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    members = [np.flatnonzero(blocks == b) for b in uniq]
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        stats.append(metric_fn(np.concatenate([members[i] for i in pick])))
    stats = np.asarray(stats, dtype=np.float64)
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))
