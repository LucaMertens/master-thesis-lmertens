import numpy as np

"""
Initial state trust
If high value: penalize deviating from state calculated in previous iteration (--> more continuous trajectory).
"""
P_x = np.diag([1, 1, 1])

"""
Smoothness / Previous parameter trust
If low value: Only slightly penalize deviating from parameters calculated in previous iteration (smoothness).
"""
# P_m = np.diag([0.5, 0.5, 0.5])
P_m = np.diag([1, 1, 1])

"""
Measurement trust
If high value: Penalize deviating from ground truth
"""
# P_w = np.diag([2, 2, 1.5])
P_w = np.diag([10, 10, 10])


# Horizon length (the same as used in the paper: https://github.com/matval/wayfaster/issues/2)
N = 8
# Time step (https://github.com/matval/wayfaster/issues/2)
# dt = 0.5  # seconds
dt = 0.1  # Sampling time of commands.
