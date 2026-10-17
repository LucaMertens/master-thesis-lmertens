"""Hyperparameters for the simple baseline traversability labeling."""

# Normalisation
max_velocity_lin: float = 1.0  # m/s
max_velocity_ang: float = 1.0  # rad/s

# Smoothing
smooth_sigma_sec: float = 1

# Sigmoid, linear traversability
k_lin: float = 75.0
v_thr_lin: float = 0.05

# Sigmoid, angular traversability
# k_ang: float = 30.0
# v_thr_ang: float = 0.15
# k_ang: float = 40.0
k_ang: float = 75
v_thr_ang: float = 0.05
