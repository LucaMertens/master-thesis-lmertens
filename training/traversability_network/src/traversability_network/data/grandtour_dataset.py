"""GrandTour dataset for the WayFASTER network.

Subclasses WayFASTER's ``Dataset`` and only replaces data loading: instead of
reading ``states.csv`` / ``images.csv`` per bag, it builds the same
``rosbag_dict`` schema from the RHE label outputs and the offline-extracted
camera data. Everything downstream (``get_states``, ``get_extrinsics``,
``read_pcloud``, ``__getitem__``) is inherited unchanged, so ``TrainingModule``
sees the exact same sample tuple as upstream WayFASTER.

Per-mission files expected in ``<DATA_ROOT>/<mission>/``:
    cam2base.npy         (4, 4)  front_center, from extract_alphasense.py
    intrinsics.npy       (3, 3)  front_center K, from extract_alphasense.py
    dlio_odom.npy        (M, 8)  t, x, y, z, qx, qy, qz, qw (base in dlio_map, ground-truth pose), from extract_dlio.py
    ground_z.npy         (M, 2)  t, gravity-aligned base-to-ground offset (~-0.55), from extract_feet.py
    image_manifest.npz   front_center structured array {ts_ns, path}
    depth_manifest.npz   structured array {ts_ns, path} (FoundationStereo output)

Per-mission files expected in ``<LABELS_ROOT>/<mission>/rhe/``:
    rhe_timestamp.npy    (N,)    RHE timestamps (seconds), the reference time grid
    rhe_trav_labels.npy  (N, >=2) columns LABEL_COLUMNS = (mu, nu)

Pose (position, orientation, action) comes from the dlio ground-truth odometry
interpolated to the RHE timestamps, NOT from rhe_states.npy (the RHE state
estimate, an optimization byproduct).

Mapping to WayFASTER's rosbag_dict:
    position       <- dlio (x, y, z) interpolated to RHE timestamps
    quaternion     <- dlio orientation slerped to RHE timestamps (x, y, z, w)
    action         <- finite-difference (v, omega) from dlio position and yaw
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

        self.bin_width = 0.05   # 20 LDS bins: resolve the low-mu tail (was 0.2 / 5 bins)
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
        self.labels_root = configs.DATASET.LABELS_ROOT
        self.resized_root = configs.DATASET.RESIZED_ROOT
        self.depth_cache_root = getattr(configs.DATASET, "DEPTH_CACHE_ROOT", None)
        self.label_columns = list(configs.MODEL.LABEL_COLUMNS)
        self.time_tolerance = configs.DATASET.TIME_TOLERANCE
        self.max_time_gap = configs.DATASET.MAX_TIME_GAP
        self.trim_stationary = configs.DATASET.TRIM_STATIONARY
        self.motion_min_speed = configs.DATASET.MOTION_MIN_SPEED
        self.motion_min_omega = configs.DATASET.MOTION_MIN_OMEGA
        self.OVERSAMPLE_FRACTION = configs.DATASET.OVERSAMPLE_FRACTION
        self.os_mu_max = configs.DATASET.OVERSAMPLE_MU_MAX
        self.os_obs_min = configs.DATASET.OVERSAMPLE_OBSERVABILITY_THRESH
        self.lds_alpha = configs.DATASET.LDS_ALPHA
        self.visibility_gate = getattr(configs.DATASET, "VISIBILITY_GATE", False)
        self.harvest_extra = getattr(configs.DATASET, "HARVEST_LOWMU_EXTRA", 0)
        self.harvest_time_window = getattr(configs.DATASET, "HARVEST_TIME_WINDOW", 0.0)

        self.rosbags = []
        self.bag_sizes = [0]

        for mission in missions:
            mission_dir = os.path.join(self.data_root, mission)
            rhe_dir = os.path.join(self.labels_root, mission, "rhe")
            for segment in self.read_mission(mission_dir, rhe_dir):
                if len(segment["image_timestamp"]) >= self.n_frames:
                    mu = np.asarray(segment["traversability"])[:, 0]
                    obs = np.asarray(segment["mu_observability"])
                    segment["lowmu_idx"] = np.flatnonzero(
                        (mu <= self.os_mu_max) & (obs >= self.os_obs_min)
                    )
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

        if self.train:
            self.indices = self._oversampled_indices()
        else:
            self.indices = list(range(self.bag_sizes[-1]))

        print("GrandTour dataset initialized!")

    def _flagged_windows(self):
        """Boolean per window (in global index order): does the near horizon contain a
        cell with mu <= OS_MU_MAX and observability >= OS_OBS_MIN that also falls
        inside the BEV grid, i.e. one _sample_path would actually score. A cell can be
        within the time horizon yet outside the xbound/ybound footprint (grid is only
        10m x 10m), in which case grid_sample masks it out and oversampling it is
        wasted. The time-based check below is only a cheap upper-bound prefilter."""
        n_flag = min(self.horizon, 300)  # only count low-mu cells in the near horizon
        xb, yb = self.grid_bounds["xbound"], self.grid_bounds["ybound"]
        flags = []
        for bag in self.rosbags:
            n_windows = len(bag["image_timestamp"]) - self.n_frames + 1
            st = np.asarray(bag["states_timestamp"])
            mu = np.asarray(bag["traversability"])[:, 0]
            observability = np.asarray(bag["mu_observability"])
            position = np.asarray(bag["position"])
            yaw = R.from_quat(bag["quaternion"]).as_euler("ZYX")[:, 0]
            qualifying = np.flatnonzero(
                (mu <= self.os_mu_max) & (observability >= self.os_obs_min)
            )

            bag_flags = np.zeros(n_windows, dtype=bool)
            if len(qualifying):
                img_ts = np.asarray(bag["image_timestamp"])[:n_windows]
                lo = np.searchsorted(st, img_ts, "left")
                hi = np.searchsorted(st, img_ts + n_flag * self.dt, "left")
                # anchor per window = last input frame's synced state (same reference
                # frame get_states/get_extrinsics use for the BEV-frame transform)
                anchor_ts = np.asarray(bag["image_timestamp"])[
                    self.n_frames - 1 : self.n_frames - 1 + n_windows
                ]
                anchor_idx = np.searchsorted(st, anchor_ts, "left")

                q_lo = np.searchsorted(qualifying, lo)
                q_hi = np.searchsorted(qualifying, hi)
                for w in np.flatnonzero(q_hi > q_lo):
                    idxs = qualifying[q_lo[w] : q_hi[w]]
                    a = anchor_idx[w]
                    heading = R.from_euler("ZYX", [yaw[a], 0, 0])
                    rel = (heading.inv().as_matrix() @ (position[idxs] - position[a]).T).T
                    inside = (
                        (rel[:, 0] >= xb[0]) & (rel[:, 0] < xb[1])
                        & (rel[:, 1] >= yb[0]) & (rel[:, 1] < yb[1])
                    )
                    if getattr(self, "visibility_gate", False):
                        # match the loss gate approximately: cells far outside the
                        # anchor camera's ~107 deg forward FOV score zero weight, so
                        # oversampling such windows would replicate dead supervision
                        inside &= np.abs(np.arctan2(rel[:, 1], rel[:, 0])) < 0.9
                    bag_flags[w] = inside.any()
            flags.append(bag_flags)
        return np.concatenate(flags)

    def _oversampled_indices(self):
        """Replicate flagged-window indices until they reach OVERSAMPLE_FRACTION of the
        index list. Deterministic, so every DDP rank builds the identical list."""
        flags = self._flagged_windows()
        all_idx = np.arange(len(flags))
        flagged, kept = all_idx[flags], all_idx[~flags]
        p = self.OVERSAMPLE_FRACTION
        # target flagged count N so that N / (len(kept) + N) == p
        n_target = int(round(p * len(kept) / (1 - p))) if p < 1 else len(flagged)
        if len(flagged) == 0 or n_target <= len(flagged):
            return all_idx.tolist()
        replicated = np.resize(flagged, n_target)  # cycles deterministically
        return np.concatenate((kept, replicated)).tolist()

    def _active_image_mask(self, img_ts, states_ts, action):
        """Keep the contiguous span of images whose forward label horizon overlaps
        motion, trimming the stationary lead-in/run-out where |v| and |omega| are
        both ~0. The horizon of a window starting at image time t runs
        [t, t + HORIZON*dt] (dataloader.get_states), so windows are informative only
        while motion falls inside that span."""
        moving = (np.abs(action[:, 0]) > self.motion_min_speed) | (
            np.abs(action[:, 1]) > self.motion_min_omega
        )
        if not moving.any():
            return np.zeros(len(img_ts), dtype=bool)
        moving_ts = states_ts[moving]
        t_first = (
            moving_ts[0] - self.horizon * self.dt
        )  # earliest window reaching motion
        t_last = (
            moving_ts[-1] + self.n_frames * self.dt
        )  # last window's trailing frames
        return (img_ts >= t_first) & (img_ts <= t_last)

    def _finite_diff_action(self, x, y, theta, timestamp):
        """(v, omega) via finite differences. v is signed forward speed."""
        vx = np.gradient(x, timestamp)
        vy = np.gradient(y, timestamp)
        v = vx * np.cos(theta) + vy * np.sin(theta)
        omega = np.gradient(np.unwrap(theta), timestamp)
        return np.column_stack((v, omega))

    def _interp_odom(self, odom, query_t):
        """Interpolate dlio ground-truth pose (M,8: t,x,y,z,qx,qy,qz,qw) onto query_t.

        Linear interp for position, slerp for orientation, both clamped to the
        odom time range. Returns (position (N,3), quaternion (N,4) in x,y,z,w order).
        """
        t = odom[:, 0]
        order = np.argsort(t)
        t = t[order]
        pos = odom[order, 1:4]
        quat_xyzw = odom[order, 4:8]
        # de-duplicate timestamps (Slerp requires strictly increasing)
        t, uniq = np.unique(t, return_index=True)
        pos = pos[uniq]
        quat_xyzw = quat_xyzw[uniq]
        clamped = np.clip(query_t, t[0], t[-1])
        position = np.column_stack([np.interp(clamped, t, pos[:, i]) for i in range(3)])
        rot = R.from_quat(quat_xyzw)
        quaternion = Slerp(t, rot)(clamped).as_quat()  # x, y, z, w
        return position, quaternion

    def read_pcloud(self, depth_image_list, cam_intrinsics_list, cam_extrinsics_list):
        """Clean full-res depth for GPU voxelisation (no CPU grid, no downsampling).

        Zeroes the top ``BORDER_CROP`` rows (the rectification black border and
        InputPadder top edge emit spurious near-depth -> phantom obstacles right
        in front of the robot) and clips to the dbound range (drops the sub-0.3 m
        halo and >8 m sky that falls outside the grid anyway). The grid itself is
        built on the GPU from this full-res depth (see TravNet.voxelize), so the
        full 1440x1080 resolution is kept for 100% voxel coverage.

        Returns the cleaned depth (T, H, W) in mm and the matching full-res
        intrinsics (T, 3, 3); the extrinsics are consumed GPU-side.
        """
        dmin, dmax = (
            self.grid_bounds["dbound"][0] * 1000,
            self.grid_bounds["dbound"][1] * 1000,
        )
        cleaned, full_K = [], []
        for depth, K in zip(depth_image_list, cam_intrinsics_list):
            depth = depth.copy()
            depth[:BORDER_CROP] = 0
            depth[(depth < dmin) | (depth > dmax)] = 0
            cleaned.append(depth.astype(np.int16))
            full_K.append(K.copy())
        return np.stack(cleaned), np.stack(full_K)

    def read_mission(self, mission_dir, rhe_dir):
        timestamp = np.load(os.path.join(rhe_dir, "rhe_timestamp.npy")).astype(
            np.float64
        )
        trav = np.load(os.path.join(rhe_dir, "rhe_trav_labels.npy"))
        mu_observability = np.load(os.path.join(rhe_dir, "rhe_mu_observability.npy"))
        nu_observability = np.load(os.path.join(rhe_dir, "rhe_nu_observability.npy"))
        cam2base = np.load(os.path.join(mission_dir, "cam2base.npy"))
        intrinsics = np.load(os.path.join(mission_dir, "intrinsics.npy"))
        odom = np.load(os.path.join(mission_dir, "dlio_odom.npy"))

        order = np.argsort(timestamp)
        timestamp = timestamp[order]
        trav = trav[order]
        mu_observability = mu_observability[order]
        nu_observability = nu_observability[order]

        # Ground-truth pose from dlio odometry (not the RHE estimate).
        odom_pos, quaternion = self._interp_odom(odom, timestamp)
        x, y = odom_pos[:, 0], odom_pos[:, 1]
        theta = R.from_quat(quaternion).as_euler("ZYX")[:, 0]
        position = odom_pos
        action = self._finite_diff_action(x, y, theta, timestamp)
        traversability = trav[:, self.label_columns]

        # Ground below the base from stance-feet kinematics (extract_feet.py)
        ground_path = os.path.join(mission_dir, "ground_z.npy")
        if os.path.exists(ground_path):
            gz = np.load(ground_path)
            ground_z = np.interp(timestamp, gz[:, 0], gz[:, 1])
        else:
            print(f"  WARNING: {ground_path} missing, assuming ground 0.55 m below base")
            ground_z = np.full_like(timestamp, -0.55)

        # ---- images (front_center) ----
        manifest = np.load(os.path.join(mission_dir, "image_manifest.npz"))
        fc = manifest["front_center"]
        img_ts = fc["ts_ns"].astype(np.float64) / 1e9
        # Prefer the local pre-resized copy (same relative paths as the manifest)
        img_root = mission_dir
        if self.resized_root:
            img_root = os.path.join(self.resized_root, os.path.basename(mission_dir))
        img_paths = [os.path.join(img_root, p) for p in fc["path"]]
        img_order = np.argsort(img_ts)
        img_ts = img_ts[img_order]
        img_paths = [img_paths[i] for i in img_order]

        # keep only frames inside the RHE state range with a future horizon
        keep = (img_ts >= timestamp[0]) & (img_ts < timestamp[-1])
        if self.trim_stationary:
            keep &= self._active_image_mask(img_ts, timestamp, action)
        if self.verbose:
            print(f"  kept {int(keep.sum())}/{len(img_ts)} frames after trimming")
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
                "ground_z": ground_z,
                "action": action,
                "traversability": traversability,
                "mu_observability": mu_observability,
                "nu_observability": nu_observability,
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
        
        depth_root = mission_dir
        if self.depth_cache_root:
            depth_root = os.path.join(self.depth_cache_root, os.path.basename(mission_dir))
        
        d_paths = [os.path.join(depth_root, p) for p in depth["path"]]
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
