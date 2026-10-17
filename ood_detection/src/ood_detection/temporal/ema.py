"""RQ6: continuous-time EMA of per-window scores (windows are unevenly spaced); half_life in seconds."""

import numpy as np


def ema_by_mission(scores, mission, timestamp, half_life=1.0):
    """NaN scores are skipped in the update, not propagated."""
    out = np.full(len(scores), np.nan, dtype=np.float64)
    decay = np.log(2) / half_life
    for m in np.unique(mission):
        idx = np.flatnonzero(mission == m)
        order = idx[np.argsort(timestamp[idx])]
        t_prev, s_prev = None, None
        for i in order:
            if np.isnan(scores[i]):
                out[i] = s_prev if s_prev is not None else np.nan
                continue
            if s_prev is None:
                s_prev = scores[i]
            else:
                dt = max(float(timestamp[i] - t_prev), 0.0)
                alpha = 1 - np.exp(-decay * dt)
                s_prev = alpha * scores[i] + (1 - alpha) * s_prev
            out[i] = s_prev
            t_prev = timestamp[i]
    return out
