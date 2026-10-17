"""GrandTour dataset: RHE labels and offline-extracted camera/odometry data on top of WayFASTER's Dataset."""

import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from ..network.dataloader import Dataset
from .sampling import speed_matched_stride, strided_indices, window_count

BORDER_CROP = 14  # top depth rows to drop (rectification border + padding artifact)

# segment entries indexed per camera frame / per RHE state, cut by time ranges
FRAME_KEYS = ("image_timestamp", "color_image", "depth_image", "intrinsics", "cam2base",
              "dino_feat_idx")
STATE_KEYS = ("states_timestamp", "position", "quaternion", "ground_z", "action",
              "traversability", "mu_observability", "nu_observability")


def normalize_ranges(ranges):
    """One pair or a list of pairs to [(t_start, t_end)]; None means an open end."""
    if ranges is None:
        return None
    pairs = ranges if isinstance(ranges[0], (list, tuple)) else [ranges]
    return [(-np.inf if a is None else float(a), np.inf if b is None else float(b))
            for a, b in pairs]


def restrict_segment(segment, ranges):
    """Sub-segments restricted to ``ranges`` (absolute seconds), frames and states both.

    Windows never bridge two ranges. ``None`` returns the segment untouched."""
    if ranges is None:
        return [segment]
    img_ts = np.asarray(segment["image_timestamp"])
    state_ts = np.asarray(segment["states_timestamp"])
    out = []
    for t0, t1 in ranges:
        s0, s1 = np.searchsorted(state_ts, [t0, t1])
        if s1 - s0 < 2:
            continue
        # a frame needs states ahead of it to be labelled
        i0, i1 = np.searchsorted(img_ts, [t0, state_ts[s1 - 1]])
        if i1 <= i0:
            continue
        out.append({
            **segment,
            **{k: segment[k][i0:i1] for k in FRAME_KEYS if k in segment},
            **{k: segment[k][s0:s1] for k in STATE_KEYS},
        })
    return out


class GrandTourDataset(Dataset):
    # class-level so a subclass can use another extraction layout
    ODOM_FILE = "dlio_odom.npy"
    CAMERA_KEY = "front_center"
    DEPTH_MAX_RANGE = None  # metres; None means trust the whole dbound range

    def __init__(self, configs, missions, transform=None, weights=None, train=False,
                 time_ranges=None):
        # mirrors WayFASTER Dataset.__init__ without CSV reading
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
        self.use_npy_sidecars = getattr(configs.DATASET, "NPY_SIDECARS", True)
        # frozen-DINO mode: samples carry cached patch features
        self.use_dino_cache = getattr(configs.MODEL, "BACKBONE", "resnet34") == "dino"
        self.dino_cache_root = getattr(configs.DATASET, "DINO_CACHE_ROOT", None)
        self.dino_model = getattr(configs.MODEL, "DINO_MODEL", None)
        self.dino_input_size = list(getattr(configs.MODEL, "DINO_INPUT_SIZE", []))
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
        self.voxel_depth_size = getattr(configs.MODEL, "VOXEL_DEPTH_SIZE", None)
        self.visibility_gate = getattr(configs.DATASET, "VISIBILITY_GATE", False)
        self.occlusion_gate = getattr(configs.DATASET, "OCCLUSION_GATE", False)
        self.harvest_extra = getattr(configs.DATASET, "HARVEST_LOWMU_EXTRA", 0)
        self.harvest_time_window = getattr(configs.DATASET, "HARVEST_TIME_WINDOW", 0.0)
        self.frame_stride = getattr(configs.DATASET, "FRAME_STRIDE", 1)
        self.window_stride = getattr(configs.DATASET, "WINDOW_STRIDE", 1)
        # ADAPTATION: per-mission FRAME_STRIDE from average speed, attached per segment
        self.per_mission_stride = getattr(configs.DATASET, "PER_MISSION_STRIDE", False)
        self.stride_target_m = getattr(configs.DATASET, "STRIDE_TARGET_M", 1.70)
        # within-mission split: a mission may supply only part of its timeline to a split
        key = "TRAIN_TIME_RANGES" if train else "VALID_TIME_RANGES"
        if time_ranges is None:
            time_ranges = getattr(configs.DATASET, key, None)
        self.time_ranges = time_ranges or {}

        self.rosbags = []
        self.bag_sizes = [0]
        # ADAPTATION: samples carry their mission index for per-mission metrics (mission_ids.py)
        self.mission_names = list(missions)
        self.emit_mission_id = True

        for m_idx, mission in enumerate(missions):
            mission_dir = os.path.join(self.data_root, mission)
            rhe_dir = os.path.join(self.labels_root, mission, "rhe")
            ranges = normalize_ranges(self.time_ranges.get(mission))
            for full in self.read_mission(mission_dir, rhe_dir):
                for segment in restrict_segment(full, ranges):
                    n_windows = window_count(
                        len(segment["image_timestamp"]), self.n_frames,
                        segment.get("frame_stride", self.frame_stride),
                    )
                    if n_windows > 0:
                        segment["mission_id"] = m_idx
                        mu = np.asarray(segment["traversability"])[:, 0]
                        obs = np.asarray(segment["mu_observability"])
                        segment["lowmu_idx"] = np.flatnonzero(
                            (mu <= self.os_mu_max) & (obs >= self.os_obs_min)
                        )
                        self.rosbags.append(segment)
                        self.bag_sizes.append(self.bag_sizes[-1] + n_windows)
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
            self.indices = strided_indices(self.bag_sizes, self.window_stride)

        print("GrandTour dataset initialized!")

    def _flagged_windows(self):
        """Per window (global index order): does the near horizon hold a low-mu cell
        inside the BEV grid that _sample_path would score?

        With the visibility gate on, that cell must also be camera-visible."""
        n_flag = min(self.horizon, 300)  # near horizon only
        xb, yb = self.grid_bounds["xbound"], self.grid_bounds["ybound"]
        use_vis = getattr(self, "visibility_gate", False)
        native = {}  # id(bag) -> (W, H) native depth dims, for the frustum test
        flags = []
        for bag in self.rosbags:
            stride = bag.get("frame_stride", self.frame_stride)
            n_windows = window_count(
                len(bag["image_timestamp"]), self.n_frames, stride
            )
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
                # anchor = synced state of the last input frame
                last = stride * (self.n_frames - 1)
                anchor_ts = np.asarray(bag["image_timestamp"])[last : last + n_windows]
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
                    if use_vis and inside.any():
                        vis = self._lowmu_window_visible(
                            bag, w, idxs[inside], rel[inside], native
                        )
                        bag_flags[w] = bool(vis.any())
                    else:
                        bag_flags[w] = bool(inside.any())
            flags.append(bag_flags)
        return np.concatenate(flags)

    def _lowmu_window_visible(self, bag, w, idxs, rel, native):
        """Per low-mu cell: camera-visible in window ``w`` (frustum test as in
        ``visible_states``, ORed over input frames). ``rel`` is in the anchor heading
        frame. Cells with unknown native dims count as visible."""
        key = id(bag)
        if key not in native:
            img = cv2.imread(bag["depth_image"][0], -1)
            native[key] = None if img is None else (img.shape[1], img.shape[0])
        if native[key] is None:
            return np.ones(len(idxs), dtype=bool)
        nat_w, nat_h = native[key]
        n, s = self.n_frames, bag.get("frame_stride", self.frame_stride)
        img_ts = [bag["image_timestamp"][w + t * s] for t in range(n)]
        cam2base = [bag["cam2base"][w + t * s].copy() for t in range(n)]
        extr = self.get_extrinsics(bag, img_ts, cam2base, False)
        if "ground_z" in bag:
            gz = rel[:, 2] + np.asarray(bag["ground_z"])[idxs]
        else:
            gz = rel[:, 2]
        pts = np.column_stack([rel[:, 0], rel[:, 1], gz])
        d0, d1 = self.grid_bounds["dbound"][0], self.grid_bounds["dbound"][1]
        vis = np.zeros(len(pts), dtype=bool)
        for t in range(min(n, len(extr))):
            K = np.asarray(bag["intrinsics"][w + t * s])
            ext = extr[t]
            cam = (pts - ext[:3, 3]) @ ext[:3, :3]
            z = cam[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = K[0, 0] * cam[:, 0] / z + K[0, 2]
                v = K[1, 1] * cam[:, 1] / z + K[1, 2]
            vis |= (z > d0) & (z < d1) & (u >= 0) & (u < nat_w) & (v >= 0) & (v < nat_h)
        return vis

    def _oversampled_indices(self):
        """Replicate flagged windows up to OVERSAMPLE_FRACTION (deterministic across DDP ranks)."""
        flags = self._flagged_windows()
        # thin by WINDOW_STRIDE first
        all_idx = np.asarray(strided_indices(self.bag_sizes, self.window_stride))
        flags = flags[all_idx]
        flagged, kept = all_idx[flags], all_idx[~flags]
        p = self.OVERSAMPLE_FRACTION
        # target flagged count N so that N / (len(kept) + N) == p
        n_target = int(round(p * len(kept) / (1 - p))) if p < 1 else len(flagged)
        if len(flagged) == 0 or n_target <= len(flagged):
            return all_idx.tolist()
        replicated = np.resize(flagged, n_target)  # cycles deterministically
        return np.concatenate((kept, replicated)).tolist()

    def _active_image_mask(self, img_ts, states_ts, action, frame_stride):
        """Mask of images whose label horizon [t, t + HORIZON*dt] overlaps motion."""
        moving = (np.abs(action[:, 0]) > self.motion_min_speed) | (
            np.abs(action[:, 1]) > self.motion_min_omega
        )
        if not moving.any():
            return np.zeros(len(img_ts), dtype=bool)
        moving_ts = states_ts[moving]
        t_first = moving_ts[0] - self.horizon * self.dt
        t_last = moving_ts[-1] + frame_stride * self.n_frames * self.dt
        return (img_ts >= t_first) & (img_ts <= t_last)

    def _finite_diff_action(self, x, y, theta, timestamp):
        """(v, omega) via finite differences. v is signed forward speed."""
        vx = np.gradient(x, timestamp)
        vy = np.gradient(y, timestamp)
        v = vx * np.cos(theta) + vy * np.sin(theta)
        omega = np.gradient(np.unwrap(theta), timestamp)
        return np.column_stack((v, omega))

    def _interp_odom(self, odom, query_t):
        """Interpolate odom (M,8: t,x,y,z,qx,qy,qz,qw) onto query_t, clamped to its time range.

        Returns position (N,3) and quaternion (N,4, x,y,z,w)."""
        t = odom[:, 0]
        order = np.argsort(t)
        t = t[order]
        pos = odom[order, 1:4]
        quat_xyzw = odom[order, 4:8]
        # Slerp needs strictly increasing times
        t, uniq = np.unique(t, return_index=True)
        pos = pos[uniq]
        quat_xyzw = quat_xyzw[uniq]
        clamped = np.clip(query_t, t[0], t[-1])
        position = np.column_stack([np.interp(clamped, t, pos[:, i]) for i in range(3)])
        rot = R.from_quat(quat_xyzw)
        quaternion = Slerp(t, rot)(clamped).as_quat()  # x, y, z, w
        return position, quaternion

    def read_pcloud(self, depth_image_list, cam_intrinsics_list, cam_extrinsics_list):
        """Clean full-res depth for GPU voxelisation.

        Zeroes the top ``BORDER_CROP`` rows (phantom near-depth from the rectification
        border) and clips to the dbound range. ``MODEL.VOXEL_DEPTH_SIZE`` resizes to a
        fixed (W, H) and scales K, for batches mixing native resolutions.

        Returns depth (T, H, W) in mm and intrinsics (T, 3, 3)."""
        dmin, dmax = (
            self.grid_bounds["dbound"][0] * 1000,
            self.grid_bounds["dbound"][1] * 1000,
        )
        if self.DEPTH_MAX_RANGE is not None:
            dmax = min(dmax, self.DEPTH_MAX_RANGE * 1000)
        target = tuple(self.voxel_depth_size) if self.voxel_depth_size else None
        cleaned, full_K = [], []
        for depth, K in zip(depth_image_list, cam_intrinsics_list):
            depth = depth.copy()
            depth[:BORDER_CROP] = 0
            depth[(depth < dmin) | (depth > dmax)] = 0
            K = K.copy()
            # resize after cleaning so the zeros survive
            if target and (depth.shape[1], depth.shape[0]) != target:
                K[:1] *= target[0] / depth.shape[1]
                K[1:2] *= target[1] / depth.shape[0]
                depth = cv2.resize(depth, target, interpolation=cv2.INTER_NEAREST)
            cleaned.append(depth.astype(np.int16))
            full_K.append(K)
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
        odom = np.load(os.path.join(mission_dir, self.ODOM_FILE))

        order = np.argsort(timestamp)
        timestamp = timestamp[order]
        trav = trav[order]
        mu_observability = mu_observability[order]
        nu_observability = nu_observability[order]

        # pose from dlio odometry, not the RHE estimate
        odom_pos, quaternion = self._interp_odom(odom, timestamp)
        x, y = odom_pos[:, 0], odom_pos[:, 1]
        theta = R.from_quat(quaternion).as_euler("ZYX")[:, 0]
        position = odom_pos
        action = self._finite_diff_action(x, y, theta, timestamp)
        traversability = trav[:, self.label_columns]

        # ADAPTATION: FRAME_STRIDE from this mission's average |v|
        if self.per_mission_stride:
            avg_speed = float(np.mean(np.abs(action[:, 0])))
            frame_stride = speed_matched_stride(
                avg_speed, self.n_frames, self.dt, self.stride_target_m, self.frame_stride,
            )
            print(
                f"  {os.path.basename(mission_dir)}: avg |v|={avg_speed:.3f} m/s "
                f"-> FRAME_STRIDE={frame_stride}"
            )
        else:
            frame_stride = self.frame_stride

        # ground below the base from stance-feet kinematics
        ground_path = os.path.join(mission_dir, "ground_z.npy")
        if os.path.exists(ground_path):
            gz = np.load(ground_path)
            ground_z = np.interp(timestamp, gz[:, 0], gz[:, 1])
        else:
            print(f"  WARNING: {ground_path} missing, assuming ground 0.55 m below base")
            ground_z = np.full_like(timestamp, -0.55)

        # images
        manifest = np.load(os.path.join(mission_dir, "image_manifest.npz"))
        fc = manifest[self.CAMERA_KEY]
        img_ts = fc["ts_ns"].astype(np.float64) / 1e9
        # prefer the pre-resized copy, fall back to mission_dir
        img_root = mission_dir
        if self.resized_root:
            candidate = os.path.join(self.resized_root, os.path.basename(mission_dir))
            if os.path.isdir(candidate):
                img_root = candidate
        img_paths = [os.path.join(img_root, p) for p in fc["path"]]
        img_order = np.argsort(img_ts)
        img_ts = img_ts[img_order]
        img_paths = [img_paths[i] for i in img_order]

        # keep only frames inside the RHE state range with a future horizon
        keep = (img_ts >= timestamp[0]) & (img_ts < timestamp[-1])
        if self.trim_stationary:
            keep &= self._active_image_mask(img_ts, timestamp, action, frame_stride)
        if self.verbose:
            print(f"  kept {int(keep.sum())}/{len(img_ts)} frames after trimming")
        img_ts = img_ts[keep]
        img_paths = [p for p, k in zip(img_paths, keep) if k]

        # depth, nearest in time
        depth_paths = self._match_depth(mission_dir, img_ts)

        # cached DINO features, exact ts_ns match
        if self.use_dino_cache:
            img_ts_ns = fc["ts_ns"].astype(np.int64)[img_order][keep]
            feat_file, feat_idx = self._match_dino_cache(mission_dir, img_ts_ns)

        # contiguous segments, so a window never bridges a gap
        breaks = np.flatnonzero(np.diff(img_ts) > self.max_time_gap) + 1
        bounds = np.concatenate(([0], breaks, [len(img_ts)]))
        segments = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if self.use_dino_cache:
                extra = {
                    "dino_feat_file": feat_file,
                    "dino_feat_file_flip": feat_file.replace("features.npy", "features_flip.npy"),
                    "dino_feat_idx": feat_idx[lo:hi],
                }
            else:
                extra = {}
            segments.append({
                **extra,
                "frame_stride": frame_stride,
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

    def _match_dino_cache(self, mission_dir, img_ts_ns):
        """Row of each kept frame in the mission's DINO feature cache; fails on a stale cache."""
        mission = os.path.basename(mission_dir)
        cache_dir = os.path.join(self.dino_cache_root, mission)
        meta_path = os.path.join(cache_dir, "meta.npz")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{meta_path} missing. Precompute with grandtour_extraction's "
                f"cache_dino_features.py for mission {mission}."
            )
        meta = np.load(meta_path, allow_pickle=True)
        assert str(meta["model"]) == self.dino_model, (
            f"{mission}: cached DINO model {meta['model']} != configured {self.dino_model}"
        )
        assert list(meta["input_size"]) == self.dino_input_size, (
            f"{mission}: cached input_size {list(meta['input_size'])} != configured {self.dino_input_size}"
        )
        ts = meta["ts_ns"]
        idx = np.searchsorted(ts, img_ts_ns)
        found = (idx < len(ts)) & (ts[np.minimum(idx, len(ts) - 1)] == img_ts_ns)
        assert found.all(), (
            f"{mission}: {int((~found).sum())}/{len(img_ts_ns)} frames missing from the "
            f"DINO cache; re-run cache_dino_features.py"
        )
        assert os.path.exists(os.path.join(cache_dir, "features_flip.npy")), (
            f"{mission}: features_flip.npy missing (old cache version); re-run cache_dino_features.py"
        )
        return os.path.join(cache_dir, "features.npy"), idx.astype(np.int64)

    def _match_depth(self, mission_dir, img_ts):
        depth_manifest_path = os.path.join(mission_dir, "depth_manifest.npz")
        if not os.path.exists(depth_manifest_path):
            raise FileNotFoundError(
                f"{depth_manifest_path} missing. Run the FoundationStereo depth generation "
                "agent (see depth_generation/prompt.md) to produce depth maps first."
            )
        dm = np.load(depth_manifest_path)
        depth = dm["depth"]
        d_ts = depth["ts_ns"].astype(np.float64) / 1e9

        depth_root = mission_dir
        if self.depth_cache_root:
            candidate = os.path.join(self.depth_cache_root, os.path.basename(mission_dir))
            if os.path.isdir(candidate):
                depth_root = candidate

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
