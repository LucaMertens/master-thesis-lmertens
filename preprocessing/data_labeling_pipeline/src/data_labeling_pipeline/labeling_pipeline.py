import datetime
import json
import pathlib

import numpy as np
from tqdm import trange

from . import hyperparameters as hp
from .dataset_loader import GrandTourLoader
from .hyperparameters import STEP_SIZE, N, P_m, P_w, P_x, dt
from .pipeline_step import is_step_done, log_step, write_step_done
from .rhe import RecedingHorizonEstimator

# Bump to invalidate labels from older versions of this step.
LABELS_VERSION = 6


def _center_on_window(a, offset):
    """Place each per-window estimate at the window center instead of its start.

    The RHE fits a constant parameter over the whole horizon, so its estimate
    describes the terrain at the window center (~(N-1)/2 steps in), not its first
    step. Shift forward so index i refers to the window centered there; pad the
    leading gap with the first estimate.
    """
    if offset == 0:
        return a
    shifted = np.empty_like(a)
    shifted[offset:] = a[:-offset]
    shifted[:offset] = a[0]
    return shifted


def run_labels(mission_path, extraction_dir, labels_dir, force: bool = False) -> None:
    DataLabelingPipeline(
        mission_path=mission_path, extraction_dir=extraction_dir, labels_dir=labels_dir
    ).process_mission(force=force)


class DataLabelingPipeline:
    def __init__(self, mission_path, extraction_dir, labels_dir):
        self.mission_path = mission_path
        self.out_path = pathlib.Path(labels_dir)
        self.dt = dt
        self.N = N
        self.loader = GrandTourLoader(
            mission_path, extracted_dir=pathlib.Path(extraction_dir)
        )

        print(
            f"Using hyperparameters:\n"
            f"P_x=\n{P_x}\n"
            f"P_m=\n{P_m}\n"
            f"P_w=\n{P_w}\n"
            f"N={N}\n"
            f"STEP_SIZE={STEP_SIZE}\n"
            f"dt={dt}"
        )

        self.rhe = RecedingHorizonEstimator(
            dt=self.dt,
            N=self.N,
            P_x=P_x,
            P_m=P_m,
            P_w=P_w,
        )

    def process_mission(self, force: bool = False):
        """Process the entire mission and save labels.

        All data is loaded in ENU convention by GrandTourLoader, which aligns
        with the unicycle kinematic model used by the RHE (see rhe.py).
        """
        out_path = self.out_path
        out_path.mkdir(parents=True, exist_ok=True)

        if not force and is_step_done(out_path, "rhe_labels", LABELS_VERSION):
            print(f"Skipping rhe_labels (up to date): {out_path}")
            return None

        with log_step("rhe_labels", out_path):
            return self._run(out_path)

    def _run(self, out_path):
        print(f"Loading data from {self.mission_path} ...")
        shared_ts, U, Z = self.loader.load_data_shared_timestamps(dt=self.dt)

        total_steps = len(shared_ts)
        print(f"Total synchronized steps: {total_steps}")

        optimized_states = np.zeros_like(Z)
        optimized_labels = np.zeros((total_steps, 3))
        optimized_mu_obs = np.zeros(total_steps)
        optimized_nu_obs = np.zeros(total_steps)
        v_means = np.zeros(total_steps)
        omega_means = np.zeros(total_steps)

        self.rhe.reset()

        print("Running Receding Horizon Estimator ...")
        # disable=None turns the bar off when stdout is not a tty (e.g. log files)
        for i in trange(
            0, total_steps - self.N + 1, STEP_SIZE, desc="RHE Progress", disable=None
        ):
            U_window = U[i : i + self.N]
            Z_window = Z[i : i + self.N]

            X_optimized, m_optimized, mu_obs, nu_obs, v_mean, omega_mean = (
                self.rhe.estimate(U_window, Z_window, step_size=STEP_SIZE)
            )

            # Fill the block with the optimized values
            optimized_states[i : i + STEP_SIZE] = X_optimized[:STEP_SIZE]
            optimized_labels[i : i + STEP_SIZE] = m_optimized[0:3]
            optimized_mu_obs[i : i + STEP_SIZE] = mu_obs
            optimized_nu_obs[i : i + STEP_SIZE] = nu_obs
            v_means[i : i + STEP_SIZE] = v_mean
            omega_means[i : i + STEP_SIZE] = omega_mean

            if i % (100 * STEP_SIZE) == 0:
                print(f"Processed step {i}/{total_steps - self.N}", flush=True)

        # Fill remaining steps if any
        last_processed_idx = ((total_steps - self.N) // STEP_SIZE) * STEP_SIZE
        remaining_start = last_processed_idx + STEP_SIZE
        if remaining_start < total_steps:
            remaining_steps = total_steps - remaining_start
            optimized_states[remaining_start:] = X_optimized[-remaining_steps:]
            optimized_labels[remaining_start:] = m_optimized[0:3]
            optimized_mu_obs[remaining_start:] = mu_obs
            optimized_nu_obs[remaining_start:] = nu_obs
            v_means[remaining_start:] = v_mean
            omega_means[remaining_start:] = omega_mean

        # rhe_states stays aligned with timestamps (per-step pose); only the
        # per-window estimates are recentered onto the window midpoint.
        offset = (self.N - 1) // 2
        optimized_labels = _center_on_window(optimized_labels, offset)
        optimized_mu_obs = _center_on_window(optimized_mu_obs, offset)
        optimized_nu_obs = _center_on_window(optimized_nu_obs, offset)
        v_means = _center_on_window(v_means, offset)
        omega_means = _center_on_window(omega_means, offset)

        print("Saving continuous labels.")
        np.save(out_path / "rhe_timestamp.npy", shared_ts)
        np.save(out_path / "rhe_states.npy", optimized_states)
        np.save(out_path / "rhe_trav_labels.npy", optimized_labels)
        np.save(out_path / "rhe_mu_observability.npy", optimized_mu_obs)
        np.save(out_path / "rhe_nu_observability.npy", optimized_nu_obs)
        np.save(out_path / "rhe_v_means.npy", v_means)
        np.save(out_path / "rhe_omega_means.npy", omega_means)

        # Save hyperparameters used for this run as JSON
        hyperparams = {
            "collection_time": datetime.datetime.now().isoformat(),
            "P_x": hp.P_x.tolist(),
            "P_m": hp.P_m.tolist(),
            "P_w": hp.P_w.tolist(),
            "N": int(hp.N),
            "STEP_SIZE": int(hp.STEP_SIZE),
            "dt": float(hp.dt),
            "v_thresh": float(hp.v_thresh),
            "w_thresh": float(hp.w_thresh),
            "max_tie_strengh": float(hp.max_tie_strengh),
        }
        with open(out_path / "rhe_hyperparameters.json", "w") as hf:
            json.dump(hyperparams, hf, indent=2)

        write_step_done(out_path, "rhe_labels", LABELS_VERSION)
        print("Mission processing complete.")
        return (
            optimized_states,
            optimized_labels,
            optimized_mu_obs,
            optimized_nu_obs,
            v_means,
            omega_means,
        )
