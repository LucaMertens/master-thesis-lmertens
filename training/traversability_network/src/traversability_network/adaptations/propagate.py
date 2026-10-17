"""Geometric label propagation for the W-RIZZ labeling tool (torch-free, full-SE(3) world poses)."""

import os
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .npy_cache import imread

# Mirrors grandtour_dataset.BORDER_CROP (duplicated to stay torch-free, pinned by test_propagate)
BORDER_CROP = 14
# anchor capped tighter than targets, its depth error hits every reprojection
Z_MIN = 0.5
Z_MAX_ANCHOR = 4.0
Z_MAX_TARGET = 6.0
# pass rate collapses past +/-50 frames (dlio drift)
WINDOW = 50
BORDER = 8
RESID_ABS, RESID_FRAC = 0.15, 0.10
SPREAD_ABS, SPREAD_FRAC = 0.15, 0.05
# anchor depth window half-width, min share with depth
ANCHOR_WIN, MIN_VALID = 2, 0.6
MIN_BASELINE, MAX_OBS, MIN_SIDE = 0.25, 24, 3
# World-space radius around the anchor bounding segment propagation;
# at or above REGION_FULL_M means the whole segment
REGION_FULL_M = 99.0
REGION_RADIUS_M = REGION_FULL_M
# Occluded when measured depth is this far in front (shared with weighting.visible_states)
OCCL_ABS, OCCL_FRAC = 0.75, 0.12


class MissionGeometry:
    """Frame index, world-frame camera poses and depth lookup for one mission."""

    def __init__(self, mission_dir, depth_root=None, camera="front_center", cache=96):
        self.mission_dir = mission_dir
        fc = np.load(os.path.join(mission_dir, "image_manifest.npz"))[camera]
        order = np.argsort(fc["ts_ns"].astype(np.int64))
        self.ts_ns = fc["ts_ns"].astype(np.int64)[order]
        self.image_paths = [os.path.join(mission_dir, p) for p in fc["path"][order]]

        self.K = np.load(os.path.join(mission_dir, "intrinsics.npy"))
        self.cam2base = np.load(os.path.join(mission_dir, "cam2base.npy"))

        t = self.ts_ns / 1e9
        odom = np.load(os.path.join(mission_dir, "dlio_odom.npy"))
        position, quat_xyzw = _interp_odom(odom, t)
        self.T_world_base = np.tile(np.eye(4), (len(t), 1, 1))
        self.T_world_base[:, :3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.T_world_base[:, :3, 3] = position
        self.T_world_cam = self.T_world_base @ self.cam2base
        self.T_cam_world = np.linalg.inv(self.T_world_cam)

        gz = np.load(os.path.join(mission_dir, "ground_z.npy"))
        self.ground_z = np.interp(t, gz[:, 0], gz[:, 1])

        self.depth_paths = _match_depth(mission_dir, depth_root, self.ts_ns)
        self._depth = lru_cache(maxsize=cache)(self._read_depth)

    def __len__(self):
        return len(self.ts_ns)

    def frame_index(self, ts_ns):
        i = int(np.searchsorted(self.ts_ns, ts_ns))
        assert i < len(self.ts_ns) and self.ts_ns[i] == ts_ns, f"{ts_ns} not in mission"
        return i

    def _read_depth(self, i):
        """Raw millimetres as uint16 (only small patches are ever converted)."""
        path = self.depth_paths[i]
        if path is None:
            return None
        d = imread(path)
        if d is None:
            return None
        d = d.copy()
        d[:BORDER_CROP] = 0
        return d

    def depth(self, i):
        return self._depth(i)


def _interp_odom(odom, query_t):
    """dlio pose (M,8: t,x,y,z,qx,qy,qz,qw) onto query_t (linear position, slerp rotation, clamped)."""
    t = odom[:, 0]
    order = np.argsort(t)
    t, pos, quat = t[order], odom[order, 1:4], odom[order, 4:8]
    t, uniq = np.unique(t, return_index=True)
    pos, quat = pos[uniq], quat[uniq]
    clamped = np.clip(query_t, t[0], t[-1])
    position = np.column_stack([np.interp(clamped, t, pos[:, i]) for i in range(3)])
    return position, Slerp(t, R.from_quat(quat))(clamped).as_quat()


def world_poses(mission_dir, times):
    """(N,4,4) world-frame base poses at arbitrary times (seconds)."""
    odom = np.load(os.path.join(mission_dir, "dlio_odom.npy"))
    position, quat = _interp_odom(odom, np.asarray(times, np.float64))
    T = np.tile(np.eye(4), (len(position), 1, 1))
    T[:, :3, :3] = R.from_quat(quat).as_matrix()
    T[:, :3, 3] = position
    return T


def _match_depth(mission_dir, depth_root, ts_ns, tolerance=0.05):
    """Nearest depth frame per image, None where none is within tolerance."""
    manifest = os.path.join(mission_dir, "depth_manifest.npz")
    if not os.path.exists(manifest):
        return [None] * len(ts_ns)
    depth = np.load(manifest)["depth"]
    root = mission_dir
    if depth_root:
        candidate = os.path.join(depth_root, os.path.basename(mission_dir))
        if os.path.isdir(candidate):
            root = candidate
    d_ts = depth["ts_ns"].astype(np.float64) / 1e9
    order = np.argsort(d_ts)
    d_ts = d_ts[order]
    paths = [os.path.join(root, p) for p in depth["path"][order]]

    img_t = ts_ns / 1e9
    idx = np.clip(np.searchsorted(d_ts, img_t), 1, len(d_ts) - 1)
    left = (img_t - d_ts[idx - 1]) <= (d_ts[idx] - img_t)
    nearest = np.where(left, idx - 1, idx)
    ok = np.abs(d_ts[nearest] - img_t) <= tolerance
    return [paths[j] if good else None for j, good in zip(nearest, ok)]


def _sample_depth(depth, u, v, win=2):
    """(median, valid fraction, p90-p10) in metres over a window of nonzero depth (uint16 mm input)."""
    h, w = depth.shape
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth[v0:v1, u0:u1]
    nz = patch[patch > 0].astype(np.float32) * 1e-3
    if nz.size == 0:
        return 0.0, 0.0, 0.0
    lo, hi = np.percentile(nz, [10, 90])
    return float(np.median(nz)), nz.size / patch.size, float(hi - lo)


def occluded(depth, u, v, z, win=2):
    """True when the depth image shows a surface clearly in front of range ``z``."""
    if depth is None:
        return False
    d, _, _ = _sample_depth(depth, int(round(u)), int(round(v)), win)
    return bool(d > 0 and d < z - max(OCCL_ABS, OCCL_FRAC * z))


def anchor_from_pixel(uv, frame, geom, mode="auto", z_max=Z_MAX_ANCHOR, win=ANCHOR_WIN,
                      min_valid=MIN_VALID, spread_frac=SPREAD_FRAC, spread_abs=SPREAD_ABS):
    """Native-resolution pixel to a world point.

    Returns a dict with ``status`` in ok / no_depth / sparse / spread / range / border,
    plus p_world, z, height_over_ground and depth diagnostics.
    """
    u, v = int(round(uv[0])), int(round(uv[1]))
    K = geom.K
    out = {"status": "border", "frame": int(frame), "uv": [float(u), float(v)],
           "z": 0.0, "n_valid": 0.0, "spread": 0.0, "mode": mode}
    depth = geom.depth(frame)
    h, w = (depth.shape if depth is not None else (1080, 1440))
    if not (BORDER <= u < w - BORDER and max(BORDER, BORDER_CROP) <= v < h - BORDER):
        return out

    ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    z, n_valid, spread = (0.0, 0.0, 0.0) if depth is None else _sample_depth(depth, u, v, win)
    out.update(n_valid=n_valid, spread=spread)

    use_plane = mode == "ground_plane" or (mode == "auto" and z <= 0)
    if not use_plane:
        if mode == "depth" and z <= 0:
            return dict(out, status="no_depth")
        if n_valid < min_valid:
            return dict(out, status="sparse")
        if spread > max(spread_abs, spread_frac * z):
            return dict(out, status="spread")
        if not (Z_MIN <= z <= z_max):
            return dict(out, status="range", z=z)
    else:
        # ray / ground plane intersection in base frame
        d_base = geom.cam2base[:3, :3] @ ray
        o_base = geom.cam2base[:3, 3]
        if d_base[2] > -1e-3:
            return dict(out, status="no_depth")
        z = float((geom.ground_z[frame] - o_base[2]) / d_base[2])
        if not (Z_MIN <= z <= z_max):
            return dict(out, status="range", z=z)
        out["mode"] = "ground_plane"

    x_cam = ray * z
    p_world = geom.T_world_cam[frame] @ np.append(x_cam, 1.0)
    p_base = geom.cam2base @ np.append(x_cam, 1.0)
    return dict(out, status="ok", z=z, p_world=p_world[:3].tolist(),
                height_over_ground=float(p_base[2] - geom.ground_z[frame]))


def propagate_batch(p_world, frames, geom, window=WINDOW, z_min=Z_MIN,
                    z_max=Z_MAX_TARGET, resid_abs=RESID_ABS, resid_frac=RESID_FRAC,
                    border=BORDER, min_baseline=MIN_BASELINE, max_obs=MAX_OBS,
                    min_side=MIN_SIDE):
    """Observations of many anchors, reading each depth image once.

    p_world (A, 3), frames (A,). Returns CSR (offsets (A+1,), obs_frame, obs_uv, obs_z, obs_resid).
    """
    p_world = np.atleast_2d(np.asarray(p_world, np.float64))
    frames = np.asarray(frames, np.int64)
    n, K = len(geom), geom.K
    homo = np.concatenate([p_world, np.ones((len(p_world), 1))], 1)

    # frustum test, grouped by frame
    by_frame = {}
    for a, f in enumerate(frames):
        lo, hi = max(0, f - window), min(n, f + window + 1)
        cam = np.einsum("jkl,l->jk", geom.T_cam_world[lo:hi], homo[a])
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * cam[:, 0] / z + K[0, 2]
            v = K[1, 1] * cam[:, 1] / z + K[1, 2]
        ok = (z >= z_min) & (z <= z_max)
        ok &= (u >= border) & (u < 1440 - border)
        ok &= (v >= max(border, BORDER_CROP)) & (v < 1080 - border)
        for j in np.flatnonzero(ok):
            by_frame.setdefault(int(lo + j), []).append((a, u[j], v[j], z[j]))

    # depth residual test, one image read per frame
    hits = [[] for _ in p_world]
    for j in sorted(by_frame):
        depth = geom.depth(j)
        if depth is None:
            continue
        for a, u, v, z in by_frame[j]:
            d, n_valid, _ = _sample_depth(depth, int(round(u)), int(round(v)))
            if d <= 0 or n_valid <= 0:
                continue
            resid = abs(d - z)
            if resid <= max(resid_abs, resid_frac * z):
                hits[a].append((j, u, v, z, resid))

    offsets, out = [0], []
    for a, obs in enumerate(hits):
        keep = _thin(sorted(obs), frames[a], geom, min_baseline, max_obs, min_side)
        out.extend(keep)
        offsets.append(len(out))
    arr = np.array(out, np.float64).reshape(-1, 5)
    return (np.array(offsets, np.int64), arr[:, 0].astype(np.int32),
            arr[:, 1:3].astype(np.float32), arr[:, 3].astype(np.float32),
            arr[:, 4].astype(np.float32))


def _thin(obs, frame, geom, min_baseline, max_obs, min_side):
    """Drop near-duplicate viewpoints, and the anchor if it lacks support on both temporal sides."""
    if not obs:
        return []
    js = np.array([o[0] for o in obs])
    if (js < frame).sum() < min_side or (js > frame).sum() < min_side:
        return []
    centers = geom.T_world_cam[js, :3, 3]
    keep, last = [], None
    for i, o in enumerate(obs):
        if last is None or np.linalg.norm(centers[i] - centers[last]) >= min_baseline:
            keep.append(o)
            last = i
        if len(keep) >= max_obs:
            break
    return keep


def propagate_point(p_world, frame, geom, **kw):
    """Frames where ``p_world`` is visible: (frames (M,), uv (M,2), z (M,), resid (M,))."""
    _, f, uv, z, resid = propagate_batch([p_world], [frame], geom, **kw)
    return f, uv, z, resid


# A region is trusted only while the segment stays one continuous surface
COHERENT_STEPS = (0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0)
COHERENT_TOL_ABS, COHERENT_TOL_FRAC = 0.12, 0.05
COHERENT_MIN_FRAC = 0.8
# Below this the plane fit is treated as degenerate and returns "no cut"
MIN_KEEP_FRAC = 0.1


def _normal(centred):
    """Plane normal of centred points via the 3x3 scatter matrix (full SVD would allocate (M, M))."""
    return np.linalg.eigh(centred.T @ centred)[1][:, 0]


def coherent_radius(idmap, seg, anchor, frame, geom, scale,
                    tol_abs=COHERENT_TOL_ABS, tol_frac=COHERENT_TOL_FRAC,
                    min_frac=COHERENT_MIN_FRAC):
    """Largest radius (m) over which the segment around the anchor stays on one plane.

    A plane is fitted to the innermost ring; the radius grows while ``min_frac`` of the
    pixels inside still sit on it. Returns None when the surface never breaks.
    """
    depth = geom.depth(frame)
    if depth is None:
        return None
    ys, xs = np.nonzero(idmap == seg)
    if len(xs) < 40:
        return None
    u, v = xs * scale, ys * scale
    z = depth[np.clip(v, 0, depth.shape[0] - 1), np.clip(u, 0, depth.shape[1] - 1)] * 1e-3
    ok = (z > Z_MIN) & (z < Z_MAX_TARGET)
    if ok.sum() < 40:
        return None
    u, v, z = u[ok], v[ok], z[ok]
    K = geom.K
    pts = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    origin = np.array([(anchor["uv"][0] - K[0, 2]) / K[0, 0], (anchor["uv"][1] - K[1, 2]) / K[1, 1], 1.0])
    origin = origin * anchor["z"]
    d = np.linalg.norm(pts - origin, axis=1)

    core = pts[d <= COHERENT_STEPS[0]]
    if len(core) < 12:
        return COHERENT_STEPS[0]
    c = core.mean(0)
    n = _normal(core - c)
    resid = np.abs((pts - c) @ n)

    last = None
    for r in COHERENT_STEPS:
        inside = d <= r
        if inside.sum() < 12:
            continue
        if (resid[inside] <= np.maximum(tol_abs, tol_frac * z[inside])).mean() < min_frac:
            return last or COHERENT_STEPS[0]
        last = r
    return None


def planar_mask(idmap, seg, anchor, frame, geom, scale, tol_abs=COHERENT_TOL_ABS,
                tol_frac=COHERENT_TOL_FRAC, connected=True):
    """Bool mask of the segment pixels on the plane fitted at the anchor, or None without depth.

    ``connected`` keeps only the component touching the anchor.
    """
    depth = geom.depth(frame)
    if depth is None:
        return None
    base = idmap == seg
    ys, xs = np.nonzero(base)
    if len(xs) < 40:
        return None
    u = np.clip(xs * scale, 0, depth.shape[1] - 1)
    v = np.clip(ys * scale, 0, depth.shape[0] - 1)
    z = depth[v, u] * 1e-3
    K = geom.K
    pts = np.column_stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z])
    cu, cv = anchor["uv"]
    origin = np.array([(cu - K[0, 2]) / K[0, 0], (cv - K[1, 2]) / K[1, 1], 1.0]) * anchor["z"]

    core = (z > Z_MIN) & (np.linalg.norm(pts - origin, axis=1) <= COHERENT_STEPS[0])
    if core.sum() < 12:
        return None
    c = pts[core].mean(0)
    n = _normal(pts[core] - c)
    keep = (z > Z_MIN) & (np.abs((pts - c) @ n) <= np.maximum(tol_abs, tol_frac * z))

    out = np.zeros_like(base)
    out[ys[keep], xs[keep]] = True
    if not connected:
        return out
    import cv2
    n_lab, lab = cv2.connectedComponents(out.astype(np.uint8), connectivity=8)
    ax, ay = int(round(cu / scale)), int(round(cv / scale))
    ax, ay = np.clip(ax, 0, out.shape[1] - 1), np.clip(ay, 0, out.shape[0] - 1)
    here = lab[ay, ax]
    out = out if here == 0 else (lab == here)   # here == 0: anchor failed the plane test
    return out if out.sum() >= MIN_KEEP_FRAC * base.sum() else None
