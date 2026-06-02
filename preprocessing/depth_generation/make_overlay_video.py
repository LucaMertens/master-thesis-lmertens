#!/usr/bin/env python
"""Render depth-overlay videos for GrandTour missions.

Colorizes depth maps and alpha-blends them over the matching RGB frames.
Writes <mission>/depth_overlay.mp4 (or --out for a single mission).
"""

import argparse
import os

import cv2
import numpy as np


def nearest_index(query_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    """For each query timestamp return the index of the nearest ref timestamp."""
    idx = np.searchsorted(ref_ts, query_ts)
    idx = np.clip(idx, 1, len(ref_ts) - 1)
    take_left = (query_ts - ref_ts[idx - 1]) <= (ref_ts[idx] - query_ts)
    return np.where(take_left, idx - 1, idx)


def colorize_depth(depth_m: np.ndarray, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    """Map metric depth to a BGR colormap; return (color, valid_mask)."""
    valid = depth_m > 0
    norm = np.clip(depth_m / max_depth, 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color, valid


def add_colorbar(frame: np.ndarray, max_depth: float) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_w, pad = 24, 60
    gradient = np.linspace(0, 255, h, dtype=np.uint8)[:, None].repeat(bar_w, axis=1)
    bar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
    out = np.zeros((h, w + pad, 3), dtype=np.uint8)
    out[:, :w] = frame
    out[:, w + 6:w + 6 + bar_w] = bar
    for frac in (0.0, 0.5, 1.0):
        y = int(np.clip(frac * h, 8, h - 4))
        cv2.putText(out, f"{frac * max_depth:.0f}m", (w + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def process_mission(mission_dir: str, out_path: str, fps: float,
                    alpha: float, max_depth: float) -> None:
    img_manifest = np.load(os.path.join(mission_dir, "image_manifest.npz"))
    depth_manifest = np.load(os.path.join(mission_dir, "depth_manifest.npz"))
    center = img_manifest["front_center"]
    depth = depth_manifest["depth"]
    if len(depth) == 0:
        print(f"[skip] {mission_dir}: no depth frames in manifest.")
        return

    order = np.argsort(center["ts_ns"])
    center = center[order]
    center_ts = center["ts_ns"].astype(np.float64)
    match = nearest_index(depth["ts_ns"].astype(np.float64), center_ts)

    writer = None
    written = 0
    for d_row, c_idx in zip(depth, match):
        rgb = cv2.imread(os.path.join(mission_dir, center["path"][c_idx]), cv2.IMREAD_COLOR)
        depth_png = cv2.imread(os.path.join(mission_dir, d_row["path"]), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth_png is None:
            continue
        if depth_png.shape[:2] != rgb.shape[:2]:
            depth_png = cv2.resize(depth_png, (rgb.shape[1], rgb.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        color, valid = colorize_depth(depth_png.astype(np.float32) / 1000.0, max_depth)
        frame = rgb.copy()
        frame[valid] = cv2.addWeighted(rgb, 1 - alpha, color, alpha, 0)[valid]
        frame = add_colorbar(frame, max_depth)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        written += 1

    if writer is not None:
        writer.release()
    print(f"Wrote {written} frames -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--missions", nargs="+", default=["/data/ETH-1"],
                   help="Mission directories to render.")
    p.add_argument("--out", default=None,
                   help="Output mp4 path (only valid with a single mission; "
                        "default <mission>/depth_overlay.mp4).")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Depth overlay opacity in [0, 1].")
    p.add_argument("--max-depth", type=float, default=20.0,
                   help="Depth (m) mapped to the top of the colormap.")
    args = p.parse_args()
    if args.out is not None and len(args.missions) != 1:
        p.error("--out requires exactly one --missions entry.")

    for m in args.missions:
        out = args.out or os.path.join(m, "depth_overlay.mp4")
        process_mission(m, out, args.fps, args.alpha, args.max_depth)


if __name__ == "__main__":
    main()
