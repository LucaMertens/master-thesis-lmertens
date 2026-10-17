#!/usr/bin/env python
"""Bird's-eye ground mosaic from run depth images, with paths overlaid."""

import argparse
import pathlib
import sys

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from shared_utils.paths import DATALOCAL_ROOT, WAYFASTER_DATASET_ROOT
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_runs import RUNS_2023_06_28, load_run
from common import parse_vec


def run_poses(cam: str, split: str, ts_folder: str, root: pathlib.Path = WAYFASTER_DATASET_ROOT):
    """Per-image pose (xyz, rotation), plus K and cam2base."""
    run_dir = root / cam / f"data_{split}" / ts_folder
    im = pd.read_csv(run_dir / "images.csv").sort_values("timestamp")
    st = pd.read_csv(run_dir / "states.csv", usecols=["timestamp", "position", "quaternion"]).sort_values("timestamp")

    ts = st["timestamp"].to_numpy()
    P = np.stack([parse_vec(x) for x in st["position"]])
    Q = np.stack([parse_vec(x) for x in st["quaternion"]])[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    ti = im["timestamp"].to_numpy()

    xyz = np.stack([np.interp(ti, ts, P[:, k]) for k in range(3)], 1)
    slerp = Rot.from_quat(Q)
    idx = np.clip(np.searchsorted(ts, ti), 1, len(ts) - 1)
    Rw = slerp[idx]  # nearest neighbour is fine at 100 Hz

    K = parse_vec(im["intrinsics"].iloc[0]).reshape(3, 3)
    C = parse_vec(im["cam2base"].iloc[0]).reshape(4, 4)
    paths = (str(run_dir) + "/" + im["image"]).to_numpy()
    dpaths = (str(run_dir) + "/" + im["depth"]).to_numpy()
    return paths, dpaths, xyz, Rw, K, C


def _correct(meta, P):
    """Apply the optimized planar pose, keep roll/pitch."""
    paths, dpaths, xyz, Rw, K, C = meta
    yaw0 = Rw.as_euler("ZYX")[:, 0]
    dz = Rot.from_euler("Z", (P[:, 2] - yaw0)[:, None])
    new_xyz = np.column_stack([P[:, 0], P[:, 1], xyz[:, 2]])
    return paths, dpaths, new_xyz, dz * Rw, K, C


FLUSH_POINTS = 8_000_000


def _flush(grid, buf, nx, ny):
    """One bincount per batch, since bincount allocates the full grid on every call."""
    if not buf:
        return
    flat = np.concatenate([b[0] for b in buf])
    col = np.concatenate([b[1] for b in buf])
    n = nx * ny
    grid["cnt"] += np.bincount(flat, minlength=n).reshape(ny, nx)
    for c in range(3):
        grid["rgb"][..., c] += np.bincount(
            flat, weights=col[:, c].astype(np.float64), minlength=n).reshape(ny, nx)
    buf.clear()


def accumulate(runs_meta, grid, extent, res, args):
    """Splat ground-height coloured points into the grid."""
    x0, y0 = extent[0], extent[2]
    nx, ny = grid["cnt"].shape[1], grid["cnt"].shape[0]
    buf, held = [], 0

    for paths, dpaths, xyz, Rw, K, C in runs_meta:
        for i in range(0, len(paths), args.frame_stride):
            dp = cv2.imread(dpaths[i], cv2.IMREAD_UNCHANGED)
            if dp is None:
                continue
            z = dp.astype(np.float32) / 1000.0
            s = args.pixel_stride
            z = z[::s, ::s]
            m = (z > 0.3) & (z < args.max_depth)
            if not m.any():
                continue
            img = cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB)[::s, ::s]

            ys, xs = np.nonzero(m)
            zz = z[ys, xs]
            u, v = xs * s, ys * s
            Pc = np.stack([(u - K[0, 2]) * zz / K[0, 0], (v - K[1, 2]) * zz / K[1, 1], zz], 1)
            Pb = Pc @ C[:3, :3].T + C[:3, 3]
            Pw = Pb @ Rw[i].as_matrix().T + xyz[i]

            g = (Pw[:, 2] > args.zmin) & (Pw[:, 2] < args.zmax)
            if not g.any():
                continue
            col = img[ys, xs][g]
            gx = ((Pw[g, 0] - x0) / res).astype(np.int64)
            gy = ((Pw[g, 1] - y0) / res).astype(np.int64)
            ok = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
            if not ok.any():
                continue
            buf.append((gy[ok] * nx + gx[ok], col[ok]))
            held += int(ok.sum())
            if held >= FLUSH_POINTS:
                _flush(grid, buf, nx, ny)
                held = 0
    _flush(grid, buf, nx, ny)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", default="zed2")
    p.add_argument("--res", type=float, default=0.2, help="Grid cell size (m).")
    p.add_argument("--zmin", type=float, default=-0.4)
    p.add_argument("--zmax", type=float, default=0.4)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--pixel-stride", type=int, default=2)
    p.add_argument("--poses", type=pathlib.Path, help="Optimized SE(2) poses to apply as a correction.")
    p.add_argument("--cache", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path)
    args = p.parse_args()

    tag = "_opt" if args.poses else ""
    args.cache = args.cache or DATALOCAL_ROOT / "wayfaster_split_videos" / f"bev{tag}.npz"
    args.out = args.out or DATALOCAL_ROOT / "wayfaster_split_videos" / f"bev_map{tag}.png"

    runs = [load_run(args.cam, s, f) for s, f in RUNS_2023_06_28]
    opt = np.load(args.poses) if args.poses else None
    if opt is not None:
        for r in runs:
            r["xy"] = opt[r["name"]][:, :2]
    allxy = np.concatenate([r["xy"] for r in runs])
    pad = 25.0
    extent = [allxy[:, 0].min() - pad, allxy[:, 0].max() + pad, allxy[:, 1].min() - pad, allxy[:, 1].max() + pad]

    if args.cache.exists():
        z = np.load(args.cache)
        rgb, cnt, extent = z["rgb"], z["cnt"], list(z["extent"])
        print(f"loaded mosaic from {args.cache}")
    else:
        nx = int((extent[1] - extent[0]) / args.res)
        ny = int((extent[3] - extent[2]) / args.res)
        grid = dict(rgb=np.zeros((ny, nx, 3), np.float64), cnt=np.zeros((ny, nx), np.int64))
        print(f"building {nx}x{ny} mosaic at {args.res} m/cell")
        meta = [run_poses(args.cam, s, f) for s, f in RUNS_2023_06_28]
        if opt is not None:
            meta = [_correct(m, opt[r["name"]]) for m, r in zip(meta, runs)]
        accumulate(meta, grid, extent, args.res, args)
        rgb, cnt = grid["rgb"], grid["cnt"]
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, rgb=rgb, cnt=cnt, extent=extent)

    filled = cnt > 0
    img = np.zeros(rgb.shape, np.uint8)
    img[filled] = np.clip(rgb[filled] / cnt[filled, None], 0, 255).astype(np.uint8)
    print(f"coverage: {100 * filled.mean():.1f}% of cells, {cnt.sum() / 1e6:.1f}M points")

    fig, ax = plt.subplots(figsize=(13, 14))
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")
    bg = np.zeros((*filled.shape, 4))
    bg[..., 3] = np.where(filled, 0.0, 1.0) * 0.06
    ax.imshow(bg, origin="lower", extent=extent, interpolation="nearest")

    for r in runs:
        c = "#3fa9ff" if r["split"] == "train" else "#ff8a1f"
        ax.plot(r["xy"][:, 0], r["xy"][:, 1], lw=1.6, color=c, alpha=0.95,
                path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    ax.plot([], [], color="#3fa9ff", lw=2, label="train")
    ax.plot([], [], color="#ff8a1f", lw=2, label="valid")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("WayFASTER 2023-06-28 zed2: ground mosaic with train/valid paths")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
