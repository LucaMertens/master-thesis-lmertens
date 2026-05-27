"""
Hyperparameters for the simple baseline traversability labeling.

Both linear and angular labels follow Frey et al. 2023:
    tau = clamp(sigmoid(-k * (err_smooth - v_thr)), 0.001, 1)

where err_smooth = gaussian_smooth(mse(v_cmd, v_actual) / max_velocity).

Normalization
--------------------------
max_velocity_lin  [m/s]:   Typical max forward/lateral speed.  Divides the
                            linear MSE so the sigmoid threshold is speed-scale
                            invariant.  tau=0.5 when RMS error ≈ sqrt(2*v_thr*max_vel).

max_velocity_ang  [rad/s]: Analogous for yaw-rate.

Smoothing
--------------------------
smooth_sigma_sec  [s]:  Standard deviation of the Gaussian smoothing kernel in
                        seconds.  Applied symmetrically (non-causal) over the full
                        mission, so there is no startup lag.  ~1-3 s is typical.

Sigmoid
--------------------------
k_lin / k_ang  [–]:  Slope.  Higher → sharper transition.  k=30 gives a
                     transition width of ≈ 4/k = 0.13 in normalised-error units.

v_thr_lin / v_thr_ang  [–]:  Normalised error at which tau = 0.5.
    With max_velocity=1.0 m/s and v_thr=0.05:
      tau=0.5 at raw MSE = 0.05 m²/s²  →  RMS error ≈ 0.32 m/s
      tau→1  for errors much below 0.05
      tau→0  for errors much above 0.05
"""

# ── Normalisation ─────────────────────────────────────────────────────────────
max_velocity_lin: float = 1.0  # m/s
max_velocity_ang: float = 1.0  # rad/s

# ── Smoothing ─────────────────────────────────────────────────────────────────
smooth_sigma_sec: float = 1  # seconds

# ── Sigmoid – linear traversability ──────────────────────────────────────────
k_lin: float = 75.0
v_thr_lin: float = 0.05

# ── Sigmoid – angular traversability ─────────────────────────────────────────
# k_ang: float = 30.0
# v_thr_ang: float = 0.15
# k_ang: float = 40.0

k_ang: float = 75
v_thr_ang: float = 0.05
