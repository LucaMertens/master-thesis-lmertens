"""RHE cost weights and window settings."""

import numpy as np

# Penalty for the first state of a window differing from the last state of the previous window
P_x = np.diag([1.5, 1.5, 1])

# Penalty for parameter change between windows (m[0], m[1] may change more than delta_theta)
P_m = np.diag([0.2, 0.2, 1.5])

# Penalty for deviation from the measured (odometry) states
P_w = np.diag([1.2, 1.2, 1])

# Horizon length: small N = responsive labels, large N = smooth labels
N = 10

# Window advance per step, needs at least one step overlap (N-1). Smaller = smoother labels, slower.
# STEP_SIZE = max(1, N - 1)
# STEP_SIZE = max(1, int(N // 2))
STEP_SIZE = 2

# Time step (https://github.com/matval/wayfaster/issues/2)
dt = 0.1

# Tying logic

# Deprecated, kept for benchmarking scripts
v_thresh = 0.2
w_thresh = 0.2

# Observability smoothstep ramp on mean command magnitude (m/s for v, rad/s for omega):
# unobservable (0) below obs_lo, fully observable (1) above obs_hi
obs_lo = 0.01
obs_hi = 0.1

# Maximum tie strength relative to P_m and P_x
max_tie_strengh = 1.3
