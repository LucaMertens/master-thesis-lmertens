"""RHE mu lookup at world points (driven strip only), for the reveal and rhe_overlap scheduling."""

import os

import numpy as np

from traversability_network.adaptations.propagate import occluded, world_poses

# Half the robot footprint: beyond this a state describes different terrain.
RADIUS = 0.5
MIN_OBS = 0.6


class RheLabels:
    """Per-mission RHE path states with their mu, in the dlio world frame."""

    def __init__(self, labels_root, mission, mission_dir):
        d = os.path.join(labels_root, mission, "rhe")
        self.ts = np.load(os.path.join(d, "rhe_timestamp.npy"))
        self.mu = np.load(os.path.join(d, "rhe_trav_labels.npy"))[:, 0]
        self.obs = np.load(os.path.join(d, "rhe_mu_observability.npy"))
        # rhe_states stays aligned with rhe_timestamp, so index i is one (pose, mu, obs) triple
        states = np.load(os.path.join(d, "rhe_states.npy"))

        ground_z = np.load(os.path.join(mission_dir, "ground_z.npy"))
        gz = np.interp(self.ts, ground_z[:, 0], ground_z[:, 1])
        T = world_poses(mission_dir, self.ts)
        under = np.einsum("nij,nj->ni", T, np.column_stack(
            [np.zeros_like(gz), np.zeros_like(gz), gz, np.ones_like(gz)]))
        # x, y from the RHE optimizer; z is the ground under the base at that time
        self.points = np.column_stack([states[:, 0], states[:, 1], under[:, 2]])

    def at(self, p_world, radius=RADIUS, min_obs=MIN_OBS):
        """Nearest path state to a world point, or None where RHE has no support."""
        d = np.linalg.norm(self.points[:, :2] - np.asarray(p_world)[:2], axis=1)
        i = int(np.argmin(d))
        if d[i] > radius or self.obs[i] < min_obs:
            return None
        return {"index": i, "mu": float(self.mu[i]), "obs": float(self.obs[i]),
                "dist": float(d[i])}

    def _project(self, geom, frame):
        homo = np.column_stack([self.points, np.ones(len(self.points))])
        cam = homo @ geom.T_cam_world[frame].T
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = geom.K[0, 0] * cam[:, 0] / z + geom.K[0, 2]
            v = geom.K[1, 1] * cam[:, 1] / z + geom.K[1, 2]
        return z, u, v

    def visible_states(self, geom, frame, z_min=0.5, z_max=4.0, border=8, min_obs=MIN_OBS):
        """(indices, uv) of path states the frame shows, with an occlusion test (as in weighting.visible_states)."""
        z, u, v = self._project(geom, frame)
        ok = (z >= z_min) & (z <= z_max) & (self.obs >= min_obs)
        ok &= (u >= border) & (u < 1440 - border) & (v >= 14 + border) & (v < 1080 - border)
        idx = np.flatnonzero(ok)
        depth = geom.depth(frame)
        idx = np.array([i for i in idx if not occluded(depth, u[i], v[i], z[i])], np.int64)
        return idx, np.column_stack([u[idx], v[idx]]).reshape(-1, 2)

    def path_pixels(self, geom, frame, z_min=0.5, z_max=6.0, border=8):
        """Driven-path pixels in this frame, in time order, with an occlusion flag per state.

        No mu or obs (the path itself is a strong prior, so such pairs are dropped from
        HDR). Stops at propagate's 6 m target cap: beyond it depth is too noisy for the occlusion test.
        """
        z, u, v = self._project(geom, frame)
        ok = (z >= z_min) & (z <= z_max)
        ok &= (u >= border) & (u < 1440 - border) & (v >= 14 + border) & (v < 1080 - border)
        idx = np.flatnonzero(ok)
        depth = geom.depth(frame)
        return [{"i": int(i), "uv": [float(u[i]), float(v[i])],
                 "occ": bool(occluded(depth, u[i], v[i], z[i]))} for i in idx]


def compare(mu_a, mu_b, deadband=0.25):
    """RHE's ordinal for a pair (+1 = b more traversable), None in the ambiguous band."""
    d = mu_b - mu_a
    if abs(d) > deadband:
        return 1 if d > 0 else -1
    if abs(d) <= deadband / 2:
        return 0
    return None
