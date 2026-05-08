import numpy as np

"""
Previous state deviation weights
Controls how much estimated state can deviate from previous state (smoothness).
"""
P_x = np.diag([0.25, 0.25, 0.25])

"""
Parameter deviation weights (m_i - m_{i-1})
Controls how much estimated parameters can change between windows
"""
P_m = np.diag([0.25, 0.25, 0.25])

"""
Ground truth deviation weights
How much to penalize deviation from ground truth.
"""
P_w = np.diag([1, 1, 1])


"""Horizon length (the same as used in the paper: https://github.com/matval/wayfaster/issues/2)"""
N = 10

"""Step size used when advancing the receding horizon window."""
STEP_SIZE = max(1, N - 1)

"""Time step (https://github.com/matval/wayfaster/issues/2)"""
dt = 0.1  # seconds


#### Tying logic

"""Linear velocity confidence threshold"""
v_thresh = 0.1

"""Angular velocity confidence threshold"""
w_thresh = 0.1

"""Exponential decay for linear velocity tie strength"""
v_decay = 0.01

"""Exponential decay for angular velocity tie strength"""
w_decay = 0.01

"""Maximum tie strength"""
lambda_max = 9
