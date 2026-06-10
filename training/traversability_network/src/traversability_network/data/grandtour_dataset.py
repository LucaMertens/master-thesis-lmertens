"""GrandTour dataset for the WayFASTER network.

Subclasses WayFASTER's ``Dataset`` and only replaces data loading: instead of
reading ``states.csv`` / ``images.csv`` per bag, it builds the same
``rosbag_dict`` schema from the RHE label outputs and the offline-extracted
camera data. Everything downstream (``get_states``, ``get_extrinsics``,
``read_pcloud``, ``__getitem__``) is inherited unchanged, so ``TrainingModule``
sees the exact same sample tuple as upstream WayFASTER.

Per-mission files expected in ``<DATA_ROOT>/<mission>/``:
    timestamp.npy        (N,)    RHE timestamps (seconds), the reference time grid
    trav_labels.npy      (N, >=2) columns LABEL_COLUMNS = (mu, nu)
    cam2base.npy         (4, 4)  from extract_extrinsics.py
    intrinsics.npy       (3, 3)  front_center K, from extract_images.py
    cpt7_odom.npy        (M, 8)  t, x, y, z, qw, qx, qy, qz (base in enu, ground-truth pose)
    image_manifest.npz   front_center structured array {ts_ns, path}
    depth_manifest.npz   structured array {ts_ns, path} (FoundationStereo output)

Pose (position, orientation, action) comes from the cpt7 ground-truth odometry
interpolated to the RHE timestamps, NOT from rhe_states.npy (the RHE state
estimate, an optimization byproduct).

Mapping to WayFASTER's rosbag_dict:
    position       <- cpt7 (x, y, 0) interpolated to RHE timestamps
    quaternion     <- cpt7 orientation slerped to RHE timestamps (w, x, y, z)
    action         <- finite-difference (v, omega) from cpt7 position and yaw
    traversability <- trav_labels[:, LABEL_COLUMNS]
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from ..network.dataloader import Dataset

BORDER_CROP = 14  # top depth rows to drop (rectification border + padding artifact)


class GrandTourDataset(Dataset):
    def __init__(self, configs, missions, transform=None, weights=None, train=False):
        # ---- mirror WayFASTER Dataset.__init__ setup (no CSV reading) ----
        print("Initializing GrandTour dataset...")
        self.transform = transform
        self.train = train
        self.dt = configs.TRAINING.DT
        self.horizon = configs.TRAINING.HORIZON
        self.image_size = configs.MODEL.INPUT_SIZE
        self.downsample = configs.MODEL.DOWNSAMPLE
        self.verbose = configs.TRAINING.VERBOSE
        self.grid_bounds = configs.MODEL.GRID_BOUNDS
        self.horiz_flip = configs.AUGMENTATIONS.HORIZ_FLIP
        self.pcloud_droput = configs.AUGMENTATIONS.PCLOUD_DROPOUT
        self.n_frames = configs.MODEL.TIME_LENGTH
        self.predict_depth = configs.MODEL.PREDICT_DEPTH
        self.max_translation_aug = configs.AUGMENTATIONS.MAX_TRANSLATION
        self.max_rotation_aug = configs.AUGMENTATIONS.MAX_ROTATION

        self.bin_width = 0.2
        import torch

        self.dtype = torch.float32

        self.map_size = (
            int(
                (self.grid_bounds["xbound"][1] - self.grid_bounds["xbound"][0])
                / self.grid_bounds["xbound"][2]
            ),
            int(
                (self.grid_bounds["ybound"][1] - self.grid_bounds["ybound"][0])
                / self.grid_bounds["ybound"][2]
            ),
        )

        # GrandTour-specific knobs
        self.data_root = configs.DATASET.DATA_ROOT
        self.label_columns = list(configs.MODEL.LABEL_COLUMNS)
        self.time_tolerance = configs.DATASET.TIME_TOLERANCE
        self.max_time_gap = configs.DATASET.MAX_TIME_GAP

        self.rosbags = []
        self.bag_sizes = [0]

        for mission in missions:
            mission_dir = os.path.join(self.data_root, mission)
            for segment in self.read_mission(mission_dir):
                if len(segment["image_timestamp"]) >= self.n_frames:
                    self.rosbags.append(segment)
                    self.bag_sizes.append(
                        self.bag_sizes[-1]
                        + len(segment["image_timestamp"])
                        - self.n_frames
                        + 1
                    )
                elif self.verbose:
                    print(f"  skipping a {mission} segment: too few frames")

        if weights is None:
            self.weights, self.bins = self.prepare_weights()
        else:
            self.weights = weights
            self.bins = np.linspace(0, 1, int(1 / self.bin_width) + 1)

        print("GrandTour dataset initialized!")

    def _finite_diff_action(self, x, y, theta, timestamp):
        """(v, omega) via finite differences. v is signed forward speed."""
        vx = np.gradient(x, timestamp)
        vy = np.gradient(y, timestamp)
        v = vx * np.cos(theta) + vy * np.sin(theta)
        omega = np.gradient(np.unwrap(theta), timestamp)
        return np.column_stack((v, omega))

    def _interp_cpt7(self, cpt7, query_t):
        """Interpolate cpt7 ground-truth pose (M,8: t,x,y,z,qw,qx,qy,qz) onto query_t.

        Linear interp for position, slerp for orientation, both clamped to the
        cpt7 time range. Returns (position (N,3), quaternion (N,4) in w,x,y,z order).
        """
        t = cpt7[:, 0]
        order = np.argsort(t)
        t = t[order]
        pos = cpt7[order, 1:4]
        quat_wxyz = cpt7[order, 4:8]
        # de-duplicate timestamps (Slerp requires strictly increasing)
        t, uniq = np.unique(t, return_index=True)
        pos = pos[uniq]
        quat_wxyz = quat_wxyz[uniq]
        clamped = np.clip(query_t, t[0], t[-1])
        position = np.column_stack([np.interp(clamped, t, pos[:, i]) for i in range(3)])
        rot = R.from_quat(quat_wxyz[:, [1, 2, 3, 0]])  # wxyz -> xyzw
        out_xyzw = Slerp(t, rot)(clamped).as_quat()
        quaternion = out_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        return position, quaternion

    def read_pcloud(self, depth_image_list, cam_intrinsics_list, cam_extrinsics_list):
        """Clean depth before voxelising.

        Zeroes the top ``BORDER_CROP`` rows (the rectification black border and
        InputPadder top edge emit spurious near-depth -> phantom obstacles right
        in front of the robot) and clips to the dbound range (drops the rest of
        the sub-0.3 m halo and >8 m sky that falls outside the grid anyway).
        """
        dmin, dmax = (
            self.grid_bounds["dbound"][0] * 1000,
            self.grid_bounds["dbound"][1] * 1000,
        )
        cleaned = []
        for depth in depth_image_list:
            depth = depth.copy()
            depth[:BORDER_CROP] = 0
            depth[(depth < dmin) | (depth > dmax)] = 0
            cleaned.append(depth)
        return super().read_pcloud(cleaned, cam_intrinsics_list, cam_extrinsics_list)

    def read_mission(self, mission_dir):
        timestamp = np.load(os.path.join(mission_dir, "timestamp.npy")).astype(
            np.float64
        )
        trav = np.load(os.path.join(mission_dir, "trav_labels.npy"))
        cam2base = np.load(os.path.join(mission_dir, "cam2base.npy"))
        intrinsics = np.load(os.path.join(mission_dir, "intrinsics.npy"))
        cpt7 = np.load(os.path.join(mission_dir, "cpt7_odom.npy"))

        order = np.argsort(timestamp)
        timestamp = timestamp[order]
        trav = trav[order]

        # Ground-truth pose from cpt7 odometry (not the RHE estimate).
        cpt7_pos, quaternion = self._interp_cpt7(cpt7, timestamp)
        x, y = cpt7_pos[:, 0], cpt7_pos[:, 1]
        theta = R.from_quat(quaternion[:, [1, 2, 3, 0]]).as_euler("zyx")[:, 0]
        position = np.column_stack((x, y, np.zeros_like(x)))
        action = self._finite_diff_action(x, y, theta, timestamp)
        traversability = trav[:, self.label_columns]

        # ---- images (front_center) ----
        manifest = np.load(os.path.join(mission_dir, "image_manifest.npz"))
        fc = manifest["front_center"]
        img_ts = fc["ts_ns"].astype(np.float64) / 1e9
        img_paths = [os.path.join(mission_dir, p) for p in fc["path"]]
        img_order = np.argsort(img_ts)
        img_ts = img_ts[img_order]
        img_paths = [img_paths[i] for i in img_order]

        # keep only frames inside the RHE state range with a future horizon
        keep = (img_ts >= timestamp[0]) & (img_ts < timestamp[-1])
        img_ts = img_ts[keep]
        img_paths = [p for p, k in zip(img_paths, keep) if k]

        # ---- depth (nearest-neighbour to front_center timestamps) ----
        depth_paths = self._match_depth(mission_dir, img_ts)

        # split into contiguous segments so a temporal window never bridges a gap
        breaks = np.flatnonzero(np.diff(img_ts) > self.max_time_gap) + 1
        bounds = np.concatenate(([0], breaks, [len(img_ts)]))
        segments = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            segments.append({
                "image_timestamp": list(img_ts[lo:hi]),
                "color_image": img_paths[lo:hi],
                "depth_image": depth_paths[lo:hi],
                "intrinsics": [intrinsics] * (hi - lo),
                "cam2base": [cam2base] * (hi - lo),
                "states_timestamp": list(timestamp),
                "position": position,
                "quaternion": quaternion,
                "action": action,
                "traversability": traversability,
            })
        return segments

    def _match_depth(self, mission_dir, img_ts):
        depth_manifest_path = os.path.join(mission_dir, "depth_manifest.npz")
        if not os.path.exists(depth_manifest_path):
            raise FileNotFoundError(
                f"{depth_manifest_path} missing. Run the FoundationStereo depth generation "
                "agent (see depth_generation/prompt.md) to produce depth maps first."
            )
        dm = np.load(depth_manifest_path)
        # depth manifest stores a single structured array under key 'depth'
        depth = dm["depth"]
        d_ts = depth["ts_ns"].astype(np.float64) / 1e9
        d_paths = [os.path.join(mission_dir, p) for p in depth["path"]]
        d_order = np.argsort(d_ts)
        d_ts = d_ts[d_order]
        d_paths = [d_paths[i] for i in d_order]

        idx = np.searchsorted(d_ts, img_ts)
        idx = np.clip(idx, 1, len(d_ts) - 1)
        left = d_ts[idx - 1]
        right = d_ts[idx]
        choose_left = (img_ts - left) <= (right - img_ts)
        nearest = np.where(choose_left, idx - 1, idx)
        deltas = np.abs(d_ts[nearest] - img_ts)
        n_bad = int(np.sum(deltas > self.time_tolerance))
        if n_bad:
            print(
                f"  WARNING: {n_bad}/{len(img_ts)} frames exceed depth time tolerance "
                f"{self.time_tolerance}s (max {deltas.max():.3f}s)"
            )
        return [d_paths[i] for i in nearest]
