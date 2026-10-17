"""Loader that feeds extracted ForEnt sequences to the RHE, mirroring GrandTourLoader.

Commands are joystick input scaled by calibrated gains (forent_extraction.gains); poses come
from the Go2 state estimator (gravity-aligned odom frame).
"""

import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

from data_labeling_pipeline.dataset_loader import (
    MissingExtractedDataError,
    fill_command_gaps_with_zero_commands,
    get_shared_timestamps,
    interpolate_unwrapped_angles,
)


class ForEntLoader:
    def __init__(self, extracted_dir):
        self.extracted_dir = pathlib.Path(extracted_dir)
        self.sequence = self.extracted_dir.name

    def _load(self, filename):
        path = self.extracted_dir / filename
        if not path.exists():
            raise MissingExtractedDataError(
                f"{path} not found. Run the forent extraction step first "
                f"(uv run python -m forent_extraction.extract --bag-dir ... "
                f"--output-dir {self.extracted_dir})."
            )
        return np.load(path)

    def load_commands(self, fill_gaps=True):
        cmd = self._load("command.npy")
        cmd_ts, cmd_linear, cmd_angular = cmd[:, 0], cmd[:, 1:4], cmd[:, 4:7]
        if fill_gaps:
            cmd_ts, cmd_linear, cmd_angular = fill_command_gaps_with_zero_commands(
                cmd_ts, cmd_linear, cmd_angular
            )
        return cmd_ts, cmd_linear, cmd_angular

    def load_odometry(self):
        odom = self._load("odom.npy")
        return odom[:, 0], odom[:, 1:4], odom[:, 4:8]

    def load_entrapments(self):
        """(N, 4) labelled spans: absolute start, end, type id, severity."""
        path = self.extracted_dir / "entrapment.npy"
        return np.load(path) if path.exists() else np.zeros((0, 4))

    def load_data_shared_timestamps(self, dt=0.1, fill_gaps=True, include_lateral_command=False):
        """Resampled (T, U, Z) for the RHE; layout as in GrandTourLoader."""
        cmd_ts, cmd_linear, cmd_angular = self.load_commands(fill_gaps=fill_gaps)
        odom_ts, odom_pos, odom_quat = self.load_odometry()
        odom_yaw = Rotation.from_quat(odom_quat).as_euler("ZYX", degrees=False)[:, 0]

        shared_ts = get_shared_timestamps(cmd_ts, odom_ts, dt=dt)
        vx = np.interp(shared_ts, cmd_ts, cmd_linear[:, 0])
        vy = np.interp(shared_ts, cmd_ts, cmd_linear[:, 1])
        omega = np.interp(shared_ts, cmd_ts, cmd_angular[:, 2])
        px = np.interp(shared_ts, odom_ts, odom_pos[:, 0])
        py = np.interp(shared_ts, odom_ts, odom_pos[:, 1])
        yaw = interpolate_unwrapped_angles(odom_ts, odom_yaw, shared_ts)

        U = (
            np.column_stack((vx, vy, omega))
            if include_lateral_command
            else np.column_stack((vx, omega))
        )
        return shared_ts, U, np.column_stack((px, py, yaw))


def run_forent_labels(extraction_dir, labels_dir, force: bool = False) -> None:
    """Run the RHE labeling on an extracted ForEnt sequence."""
    from data_labeling_pipeline.labeling_pipeline import DataLabelingPipeline

    DataLabelingPipeline(
        mission_path=extraction_dir,
        extraction_dir=extraction_dir,
        labels_dir=labels_dir,
        loader=ForEntLoader(extraction_dir),
    ).process_mission(force=force)
