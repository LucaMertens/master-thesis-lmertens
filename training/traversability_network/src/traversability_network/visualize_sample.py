"""Sanity overlay: depth, future path and BEV occupancy for training samples.

Renders, per anchor frame, exactly the geometry the loss sees (reuses the
dataset's get_states / get_extrinsics / read_pcloud):
  left  - raw RGB with depth blended in and the future path reprojected
  right - BEV z-height from read_pcloud with the future path drawn on top

Static frames:
    uv run --project training/traversability_network python -m \
        traversability_network.visualize_sample --mission ETH-1 --indices 800 1500

Video:
    uv run --project training/traversability_network python -m \
        traversability_network.visualize_sample --mission ETH-1 \
        --video ETH-1_overlay.mp4 --stride 2 --fps 15
"""

import argparse
import bisect
import os
from functools import lru_cache

import cv2
import numpy as np

from .config import get_cfg
from .data.grandtour_dataset import GrandTourDataset
from .network.dataloader import Dataset

CFG = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "grandtour.yaml")

GREEN = (80, 255, 80)
RED = (60, 60, 255)
DEPTH_CMAP = cv2.COLORMAP_TURBO  # left: range, rainbow (conventional)
HEIGHT_CMAP = cv2.COLORMAP_MAGMA  # right: height, sequential (distinct from depth)


@lru_cache(maxsize=64)
def _imread(path, flag):
    """Cached read; consecutive samples share frames (overlapping windows)."""
    return cv2.imread(path, flag)


def _locate(ds, idx):
    bag = bisect.bisect_right(ds.bag_sizes, idx) - 1
    return ds.rosbags[bag], idx - ds.bag_sizes[bag]


def _bev_params(ds):
    gb = ds.grid_bounds
    origin = (gb["xbound"][1] / gb["xbound"][2], gb["ybound"][1] / gb["ybound"][2])
    res = (gb["xbound"][2], gb["ybound"][2])
    return origin, res


def _occupancy(ds, depths, K_list, extr, scale):
    """Z-resolved BEV hit-grid (Z, X, Y). Downsampling depth speeds up read_pcloud."""
    if scale != 1.0:
        ds_depths, ds_K = [], []
        for d, K in zip(depths, K_list):
            h, w = d.shape
            ds_depths.append(
                cv2.resize(
                    d, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST
                )
            )
            K2 = K.copy()
            K2[:2] *= scale
            ds_K.append(K2)
        depths, K_list = ds_depths, ds_K
    # GrandTourDataset.read_pcloud now builds the grid GPU-side (see TravNet.voxelize)
    # and returns (depth, K) instead; use the base class's CPU grid for this sanity view.
    grid = Dataset.read_pcloud(ds, depths, K_list, extr)  # (n_frames, Z, X, Y)
    return grid.max(axis=0)  # (Z, X, Y)


def _geometry(ds, idx, bev_scale):
    """All geometry for one anchor frame (no augmentation), with raw images kept."""
    rosbag, data_idx = _locate(ds, idx)
    ts, color_paths, depths, K_list, cam2base = [], [], [], [], []
    for t in range(ds.n_frames):
        ts.append(rosbag["image_timestamp"][data_idx + t])
        color_paths.append(rosbag["color_image"][data_idx + t])
        depths.append(_imread(rosbag["depth_image"][data_idx + t], -1))
        K_list.append(rosbag["intrinsics"][data_idx + t].copy())
        cam2base.append(rosbag["cam2base"][data_idx + t].copy())

    states, _, _, _ = ds.get_states(rosbag, ts, np.eye(4), False)
    extrinsics = ds.get_extrinsics(rosbag, ts, cam2base, False)
    occ = _occupancy(ds, depths, K_list, extrinsics, bev_scale)

    path = states[np.isfinite(states[:, 0])]
    color_bgr = _imread(color_paths[-1], -1)
    return path, occ, color_bgr, depths[-1], K_list[-1], extrinsics[-1]


def _project_path(path, K, extrinsics):
    """Anchor-frame path (x, y, 0) into the anchor camera image (inverse of read_pcloud)."""
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]
    pts = np.column_stack((path[:, 0], path[:, 1], np.zeros(len(path))))
    cam = (R.T @ (pts - t).T).T
    front = cam[:, 2] > 1e-3
    pix = (K @ cam[front].T).T
    return pix[:, :2] / pix[:, 2:3]


def _blend_depth(color_bgr, depth, max_depth):
    d = depth.astype(np.float32) * 1e-3
    valid = (d > 0) & (d <= max_depth)
    cmap = cv2.applyColorMap(
        (np.clip(d / max_depth, 0, 1) * 255).astype(np.uint8), DEPTH_CMAP
    )
    out = color_bgr.copy()
    out[valid] = (0.5 * color_bgr[valid] + 0.5 * cmap[valid]).astype(np.uint8)
    return out


def _colorbar(img, cmap, title, vmin, vmax, x, mark_zero=False):
    """Draw a labelled vertical colorbar at column x (top = vmax, bottom = vmin)."""
    h = img.shape[0]
    bar_h, bar_w, y0 = int(h * 0.45), 16, int(h * 0.12)
    grad = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None].repeat(bar_w, 1)
    img[y0 : y0 + bar_h, x : x + bar_w] = cv2.applyColorMap(grad, cmap)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    cv2.putText(img, title, (x - 4, y0 - 8), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    def tick(val):
        y = int(y0 + (vmax - val) / (vmax - vmin) * bar_h)
        cv2.line(img, (x, y), (x + bar_w, y), (255, 255, 255), 1)
        cv2.putText(
            img,
            f"{val:.1f}",
            (x + bar_w + 4, y + 4),
            font,
            fs,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    tick(vmax)
    tick(vmin)
    if mark_zero and vmin < 0 < vmax:
        tick(0.0)


def _polyline(img, xy, color):
    if len(xy):
        cv2.polylines(
            img, [xy.astype(np.int32).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA
        )


def compose(ds, idx, height, max_depth, bev_scale):
    """Two-panel BGR frame: camera+depth+path | BEV height-grid+path."""
    path, zgrid, color_bgr, depth, K, extr = _geometry(ds, idx, bev_scale)
    origin, res = _bev_params(ds)

    # left: camera + depth blend + reprojected path
    cam = _blend_depth(color_bgr, depth, max_depth)
    s = height / cam.shape[0]
    cam = cv2.resize(cam, (int(cam.shape[1] * s), height))
    uv = _project_path(path, K, extr) * s
    inside = (
        (uv[:, 0] >= 0)
        & (uv[:, 0] < cam.shape[1])
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < cam.shape[0])
    )
    _polyline(cam, uv[inside], GREEN)
    cv2.putText(
        cam,
        f"idx {idx}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    _colorbar(cam, DEPTH_CMAP, "depth m", 0.0, max_depth, cam.shape[1] - 70)

    # right: BEV coloured by highest physical return per cell (read_pcloud z-index is inverted)
    nz = zgrid.shape[0]
    zb = ds.grid_bounds["zbound"]
    z_levels = np.arange(nz)[:, None, None]
    phys = zb[0] + (nz - 1 - z_levels) * zb[2]  # physical height per z-index
    h_top = np.where(zgrid > 0, phys, -np.inf).max(0)  # highest surface per column
    has_hit = np.isfinite(h_top)
    vmin, vmax = zb[0], zb[0] + (nz - 1) * zb[2]
    norm = np.clip((h_top - vmin) / (vmax - vmin), 0, 1)
    bev = cv2.applyColorMap((norm * 255).astype(np.uint8), HEIGHT_CMAP)
    bev[~has_hit] = (40, 40, 40)
    sb = height / zgrid.shape[1]
    bev = cv2.resize(bev, (height, height), interpolation=cv2.INTER_NEAREST)
    u = np.round(origin[0] - path[:, 0] / res[0])
    v = np.round(origin[1] - path[:, 1] / res[1])
    in_map = (u >= 0) & (u < zgrid.shape[1]) & (v >= 0) & (v < zgrid.shape[2])
    _polyline(bev, np.column_stack((v, u)) * sb, GREEN)
    cv2.circle(bev, (int(origin[1] * sb), int(origin[0] * sb)), 6, RED, -1)
    cv2.putText(
        bev,
        f"in map {int(in_map.sum())}/{len(path)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bev,
        "green=path  red=robot  dark=no return",
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    _colorbar(bev, HEIGHT_CMAP, "height m", vmin, vmax, height - 70, mark_zero=True)

    return cv2.hconcat([cam, bev])


def render_video(ds, out, start, end, stride, fps, height, max_depth, bev_scale):
    end = len(ds) if end is None else min(end, len(ds))
    indices = list(range(start, end, stride))

    def _frame(idx):
        f = compose(ds, idx, height, max_depth, bev_scale)
        f = f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2]  # even dims for mp4v
        return np.ascontiguousarray(f, dtype=np.uint8)

    first = _frame(indices[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer.write(first)
    for n, idx in enumerate(indices[1:], 1):
        writer.write(_frame(idx))
        if n % 50 == 0:
            print(f"  {n}/{len(indices)} frames")
    writer.release()
    print(f"wrote {out}  ({len(indices)} frames @ {fps} fps)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mission", default="ETH-1")
    p.add_argument(
        "--indices", type=int, nargs="+", default=[800], help="static-frame mode"
    )
    p.add_argument(
        "--video", default=None, help="output .mp4 path (enables video mode)"
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument(
        "--bev-scale", type=float, default=0.5, help="depth downsample for the hit-grid"
    )
    p.add_argument("--out", default="sample_overlays", help="dir for static frames")
    args = p.parse_args()

    cfg = get_cfg(CFG)
    ds = GrandTourDataset(cfg, [args.mission], train=False)
    print(f"{args.mission}: {len(ds)} samples")

    if args.video:
        render_video(
            ds,
            args.video,
            args.start,
            args.end,
            args.stride,
            args.fps,
            args.height,
            args.max_depth,
            args.bev_scale,
        )
    else:
        os.makedirs(args.out, exist_ok=True)
        for idx in args.indices:
            if 0 <= idx < len(ds):
                frame = compose(ds, idx, args.height, args.max_depth, args.bev_scale)
                path_out = os.path.join(args.out, f"overlay_{idx:06d}.png")
                cv2.imwrite(path_out, frame)
                print(f"  wrote {path_out}")
            else:
                print(f"  skip idx {idx} (out of range)")


if __name__ == "__main__":
    main()
