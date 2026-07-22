"""Extra low-mu states harvested from anywhere in a bag, not only the 30s future path."""
import numpy as np
from scipy.spatial.transform import Rotation as R

from .geometry import ground_height


def harvest_lowmu(rosbag_dict, timestamps, curr_position, heading_rot, aug, horizontal_flip,
                   harvest_extra, harvest_time_window, grid_bounds):
    """Low-mu states outside this window's horizon, in the anchor frame and filtered to the
    grid; the visibility gate downstream still decides whether an input frame sees them."""
    st = np.asarray(rosbag_dict["states_timestamp"])
    idx = rosbag_dict["lowmu_idx"]
    on_path = (st[idx] >= timestamps[0]) & (st[idx] <= timestamps[-1])
    idx = idx[~on_path]
    if harvest_time_window > 0:
        in_window = np.abs(st[idx] - timestamps[0]) <= harvest_time_window
        idx = idx[in_window]
    if len(idx) == 0:
        return None
    position = np.asarray(rosbag_dict["position"])[idx]
    position = (heading_rot.inv().as_matrix() @ (position - curr_position).T).T
    if horizontal_flip:
        position[:, 1] = -position[:, 1]
    position = (aug[:3, :3] @ position.T).T + aug[:3, 3]
    xb, yb = grid_bounds["xbound"], grid_bounds["ybound"]
    inside = (
        (position[:, 0] >= xb[0]) & (position[:, 0] < xb[1])
        & (position[:, 1] >= yb[0]) & (position[:, 1] < yb[1])
    )
    idx, position = idx[inside], position[inside]
    if len(idx) == 0:
        return None
    if len(idx) > harvest_extra:
        pick = np.random.choice(len(idx), harvest_extra, replace=False)
        idx, position = idx[pick], position[pick]
    yaw = R.from_quat(np.asarray(rosbag_dict["quaternion"])[idx]).as_euler("ZYX")[:, :1]
    if horizontal_flip:
        yaw = -yaw
    yaw = yaw + np.arctan2(aug[1, 0], aug[0, 0])
    action = np.asarray(rosbag_dict["action"])[idx]
    ground = ground_height(rosbag_dict, idx, position)
    states = np.hstack((position[:, :2], yaw, action, ground[:, None]))
    return (
        states,
        np.asarray(rosbag_dict["traversability"])[idx],
        np.asarray(rosbag_dict["mu_observability"])[idx],
        np.asarray(rosbag_dict["nu_observability"])[idx],
    )
