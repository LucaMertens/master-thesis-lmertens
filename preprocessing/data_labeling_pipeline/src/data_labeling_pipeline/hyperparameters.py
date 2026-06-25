import numpy as np

"""
Previous state deviation weights
We estimate the first state of the current window twice: once as the last state of the previous window, and once as the first state of the current window.
This term penalizes deviation between these two, encouraging smoothness.
"""
P_x = np.diag([1.5, 1.5, 1])

"""
Parameter deviation ^ (m_i - m_{i-1})
Controls how much estimated parameters can change between windows.
Linear traversability labels (m[0], m[1]) are allowed to change more than delta_theta, which should be more stable.
"""
P_m = np.diag([0.2, 0.2, 1.5])

"""
Ground truth deviation weights
How much to penalize deviation from ground truth.
"""
P_w = np.diag([1.2, 1.2, 1])


"""Horizon length. Trade-off between label responsiveness (smaller N) and smoothness (larger N)."""
N = 10

"""Step size used when advancing the receding horizon window.
Needs at least one step overlap (N-1).

Smaller step size --> more optimization windows and smoother labels, but also more computation time.
"""
# STEP_SIZE = max(1, N - 1)
# STEP_SIZE = max(1, int(N // 2))
STEP_SIZE = 2

"""Time step (https://github.com/matval/wayfaster/issues/2)"""
dt = 0.1

#### Tying logic

"""Deprecated: Linear velocity observability threshold (legacy, kept for benchmarking scripts)."""
v_thresh = 0.2

"""Deprecated: Angular velocity observability threshold (legacy, kept for benchmarking scripts)."""
w_thresh = 0.2

"""Observability smoothstep ramp on the mean command magnitude (m/s for v, rad/s for omega).
A parameter is unobservable (0) below obs_lo and fully observable (1) above obs_hi."""
obs_lo = 0.01
obs_hi = 0.1

"""Maximum tie strength (how important the tying is, compared to P_m and P_x)."""
max_tie_strengh = 1.3
