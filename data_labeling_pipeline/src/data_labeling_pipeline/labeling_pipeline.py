import pathlib

import numpy as np

from .dataset_loader import GrandTourLoader
from .hyperparameters import N, P_m, P_w, P_x, dt
from .rhe import RecedingHorizonEstimator


class DataLabelingPipeline:
    def __init__(self, mission_path):
        self.mission_path = mission_path
        self.dt = dt
        self.N = N
        self.loader = GrandTourLoader(mission_path)

        print(
            f"Using hyperparameters:\n"
            f"P_x=\n{P_x}\n"
            f"P_m=\n{P_m}\n"
            f"P_w=\n{P_w}\n"
            f"N={N}\n"
            f"dt={dt}"
        )

        self.rhe = RecedingHorizonEstimator(
            dt=self.dt,
            N=self.N,
            P_x=P_x,
            P_m=P_m,
            P_w=P_w,
        )

    def process_mission(self, output_dir):
        """Process the entire mission and save labels.

        All data is loaded in ENU convention by GrandTourLoader, which aligns
        with the unicycle kinematic model used by the RHE (see rhe.py).
        """
        out_path = pathlib.Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"Loading data from {self.mission_path} ...")
        shared_ts, U, Z = self.loader.load_data_shared_timestamps(dt=self.dt)

        total_steps = len(shared_ts)
        print(f"Total synchronized steps: {total_steps}")

        optimized_states = np.zeros_like(Z)
        optimized_labels = np.zeros((total_steps, 3))

        self.rhe.reset()

        print("Running Receding Horizon Estimator ...")
        for i in range(total_steps - self.N + 1):
            U_window = U[i : i + self.N]
            Z_window = Z[i : i + self.N]

            X_opt, m_opt = self.rhe.estimate(U_window, Z_window)

            optimized_states[i] = X_opt[0]
            optimized_labels[i] = m_opt[0:3]

            if i % 100 == 0:
                print(f"Processed step {i}/{total_steps - self.N}")

        for j in range(1, self.N):
            idx = total_steps - self.N + j
            if idx < total_steps:
                optimized_states[idx] = X_opt[j]
                optimized_labels[idx] = m_opt[0:3]

        print("Saving continuous labels ...")
        np.save(out_path / "timestamp.npy", shared_ts)
        np.save(out_path / "x_opt.npy", optimized_states)
        np.save(out_path / "trav_labels.npy", optimized_labels)

        print("Mission processing complete.")
        return optimized_states, optimized_labels
