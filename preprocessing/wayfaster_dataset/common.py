"""Run list, colors and parsing helpers shared by the WayFASTER scripts."""

import re

import numpy as np

RUNS_2023_06_28 = [
    ("train", "ts_2023_06_28_20h12m04s"),
    ("train", "ts_2023_06_28_20h12m27s"),
    ("train", "ts_2023_06_28_20h12m37s"),
    ("train", "ts_2023_06_28_20h28m27s"),
    ("train", "ts_2023_06_28_20h29m17s"),
    ("valid", "ts_2023_06_28_20h29m58s"),
    ("train", "ts_2023_06_28_20h47m38s"),
    ("valid", "ts_2023_06_28_20h48m37s"),
    ("valid", "ts_2023_06_28_21h04m14s"),
]

COLORS = {"train": (0, 170, 0), "valid": (0, 100, 240)}  # BGR


def parse_vec(s: str) -> np.ndarray:
    return np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)])


def nearest_frame(run: dict, t: float, tol: float):
    """Closest frame path to t, or None if outside the run or in a gap wider than tol."""
    ts = run["t"]
    if t < ts[0] - tol or t > ts[-1] + tol:
        return None
    i = min(max(np.searchsorted(ts, t), 1), len(ts) - 1)
    j = i - 1 if abs(t - ts[i - 1]) <= abs(ts[i] - t) else i
    return run["paths"][j] if abs(t - ts[j]) <= tol else None
