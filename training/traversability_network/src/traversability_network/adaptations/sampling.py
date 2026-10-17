"""Temporal sampling of input windows: intra-window frame stride and window start stride."""

import numpy as np


def frame_offsets(n_frames, stride):
    """Offsets of the stacked input frames from the window start."""
    return stride * np.arange(n_frames)


def window_count(n_timestamps, n_frames, stride):
    """Windows fitting in a segment of n_timestamps frames, 0 if it is too short."""
    return max(0, n_timestamps - stride * (n_frames - 1))


def speed_matched_stride(avg_speed, n_frames, dt, target_m, fallback):
    """FRAME_STRIDE so the window span (n_frames-1)*stride*dt covers target_m at avg_speed."""
    if avg_speed < 1e-3:
        return fallback
    stride = target_m / ((n_frames - 1) * dt * avg_speed)
    return max(1, round(stride))


def strided_indices(bag_sizes, stride):
    """Global window indices whose start is a multiple of stride inside its own bag."""
    return [i for lo, hi in zip(bag_sizes[:-1], bag_sizes[1:]) for i in range(lo, hi, stride)]
