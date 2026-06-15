"""
Simple baseline traversability labeling based on Frey et al. 2023.

Pipeline (linear and angular share the same structure):

    mse_lin  = 0.5 * ((vx_cmd-vx)^2 + (vy_cmd-vy)^2)   [m^2/s^2]
    err_lin  = mse_lin / max_velocity_lin                [normalised]
    err_smooth = gaussian_filter1d(err_lin, sigma)       [non-causal, no startup lag]
    tau_lin  = clamp(sigmoid(-k_lin * (err_smooth - v_thr_lin)), 0.001, 1)

    mse_ang  = (omega_cmd - omega)^2
    err_ang  = mse_ang / max_velocity_ang
    ...same smoothing and sigmoid...

Actual velocities are obtained by numerically differentiating the IE odometry
pose and rotating into the body frame.

A symmetric (non-causal) Gaussian smoother is used instead of a causal Kalman
filter because we process complete missions offline.  A causal filter initialised
at zero would produce artificially high traversability scores for the first tens
of seconds regardless of actual tracking quality.
"""

import datetime
import json
import pathlib

import numpy as np
from data_labeling_pipeline.dataset_loader import GrandTourLoader
from scipy.ndimage import gaussian_filter1d

from . import hyperparameters as hp


def _smooth(signal: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Symmetric Gaussian smoothing over a batch signal.

    Non-causal: uses future samples, which is correct for offline processing
    and avoids the startup-lag problem of causal filters.
    """
    return gaussian_filter1d(signal.astype(float), sigma=max(sigma_samples, 1e-6))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def traversability_from_velocity_error(
    v_error: np.ndarray,
    k: float,
    v_thr: float,
) -> np.ndarray:
    """Map a smoothed, normalised velocity error to a traversability score.

    tau = sigmoid(-k * (v_error - v_thr))

    High v_error → low tau (not traversable).
    Low v_error  → high tau (traversable).
    tau = 0.5 when v_error = v_thr.
    """
    return _sigmoid(-k * (v_error - v_thr))


def _body_frame_velocities(
    px: np.ndarray,
    py: np.ndarray,
    yaw: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute body-frame velocities from numerical differentiation of pose.
    Returns: (v_x, v_y, omega) each of shape (N,).
    """
    vx_global = np.gradient(px, timestamps)
    vy_global = np.gradient(py, timestamps)
    omega = np.gradient(np.unwrap(yaw), timestamps)

    v_x = vx_global * np.cos(yaw) + vy_global * np.sin(yaw)
    v_y = -vx_global * np.sin(yaw) + vy_global * np.cos(yaw)

    return v_x, v_y, omega


class SimpleBaselineLabeler:
    def __init__(
        self,
        mission_path,
        k_lin: float = hp.k_lin,
        v_thr_lin: float = hp.v_thr_lin,
        k_ang: float = hp.k_ang,
        v_thr_ang: float = hp.v_thr_ang,
        max_velocity_lin: float = hp.max_velocity_lin,
        max_velocity_ang: float = hp.max_velocity_ang,
        smooth_sigma_sec: float = hp.smooth_sigma_sec,
        dt: float = 0.1,
    ):
        self.mission_path = pathlib.Path(mission_path)
        self.k_lin = k_lin
        self.v_thr_lin = v_thr_lin
        self.k_ang = k_ang
        self.v_thr_ang = v_thr_ang
        self.max_velocity_lin = max_velocity_lin
        self.max_velocity_ang = max_velocity_ang
        self.smooth_sigma_sec = smooth_sigma_sec
        self.dt = dt
        self.loader = GrandTourLoader(mission_path)

    def process_mission(self, output_dir) -> dict:
        """Compute simple baseline traversability labels and save to disk."""
        out_path = pathlib.Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        shared_ts, U, Z = self.loader.load_data_shared_timestamps(
            dt=self.dt, include_lateral_command=True
        )

        vx_cmd = U[:, 0]
        vy_cmd = U[:, 1]
        yaw_cmd = U[:, 2]

        px = Z[:, 0]
        py = Z[:, 1]
        yaw = Z[:, 2]

        vx_actual, vy_actual, yaw_actual = _body_frame_velocities(
            px, py, yaw, shared_ts
        )

        # MSE error normalised by max velocity — keeps the sigmoid threshold
        # scale-invariant across different commanded speeds.
        v_err_lin_raw = (
            0.5
            * ((vx_actual - vx_cmd) ** 2 + (vy_actual - vy_cmd) ** 2)
            / self.max_velocity_lin
        )
        v_err_ang_raw = (yaw_cmd - yaw_actual) ** 2 / self.max_velocity_ang

        # Gaussian smoothing over the full mission (non-causal, no startup lag).
        sigma_samples = self.smooth_sigma_sec / self.dt
        v_err_lin = _smooth(v_err_lin_raw, sigma_samples)
        v_err_ang = _smooth(v_err_ang_raw, sigma_samples)
        # v_err_lin = v_err_lin_raw
        # v_err_ang = v_err_ang_raw

        tau_lin = np.clip(
            traversability_from_velocity_error(v_err_lin, self.k_lin, self.v_thr_lin),
            0.001,
            1.0,
        )
        tau_ang = np.clip(
            traversability_from_velocity_error(v_err_ang, self.k_ang, self.v_thr_ang),
            0.001,
            1.0,
        )

        np.save(out_path / "rhe_timestamp.npy", shared_ts)
        np.save(out_path / "simple_trav_lin.npy", tau_lin)
        np.save(out_path / "simple_trav_ang.npy", tau_ang)
        np.save(out_path / "simple_v_error_lin.npy", v_err_lin)
        np.save(out_path / "simple_v_error_ang.npy", v_err_ang)
        np.save(out_path / "simple_v_error_lin_raw.npy", v_err_lin_raw)
        np.save(out_path / "simple_v_error_ang_raw.npy", v_err_ang_raw)

        hyperparams = {
            "collection_time": datetime.datetime.now().isoformat(),
            "k_lin": self.k_lin,
            "v_thr_lin": self.v_thr_lin,
            "k_ang": self.k_ang,
            "v_thr_ang": self.v_thr_ang,
            "max_velocity_lin": self.max_velocity_lin,
            "max_velocity_ang": self.max_velocity_ang,
            "smooth_sigma_sec": self.smooth_sigma_sec,
            "dt": self.dt,
        }
        with open(out_path / "simple_hyperparameters.json", "w") as hf:
            json.dump(hyperparams, hf, indent=2)

        print(f"Simple baseline labels saved to {out_path}")
        return {
            "timestamp": shared_ts,
            "vx_cmd": vx_cmd,
            "vy_cmd": vy_cmd,
            "yaw_cmd": yaw_cmd,
            "vx_actual": vx_actual,
            "vy_actual": vy_actual,
            "yaw_actual": yaw_actual,
            "v_err_lin_raw": v_err_lin_raw,
            "v_err_ang_raw": v_err_ang_raw,
            "v_err_lin": v_err_lin,
            "v_err_ang": v_err_ang,
            "tau_lin": tau_lin,
            "tau_ang": tau_ang,
        }
