#!/usr/bin/env python
"""Odometry-chained lanes, each with a heading-up navigation minimap."""

import argparse
import pathlib
import sys

import cv2
import imageio.v2 as iio
import numpy as np
from shared_utils.paths import DATALOCAL_ROOT

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_runs import RUNS_2023_06_28, load_run
from common import COLORS
from make_chained_tile_video import build_lanes

IDLE_COLOR = (40, 40, 40)
FAINT = (70, 70, 70)
BANNER_H = 20
GLOBAL_BAR_H = 24
MAP_H = 240
DRAW_STEP = 3  # polyline subsampling


def draw_map(runs, active, pose, w, h, mpp):
    """Heading-up minimap centred on pose; active route bold, others faint."""
    img = np.full((h, w, 3), 18, np.uint8)
    cx, cy = w // 2, int(h * 0.68)
    x, y, yaw = pose
    ef = np.array([np.cos(yaw), np.sin(yaw)])
    el = np.array([-np.sin(yaw), np.cos(yaw)])

    def to_map(P):
        d = P - np.array([x, y])
        along, lat = d @ ef, d @ el
        return np.stack([cx - lat / mpp, cy - along / mpp], 1).astype(np.int32)

    for r in runs:
        if r is active:
            continue
        cv2.polylines(img, [to_map(r["xy"][::DRAW_STEP])], False, FAINT, 1, cv2.LINE_AA)
    if active is not None:
        cv2.polylines(img, [to_map(active["xy"][::DRAW_STEP])], False, COLORS[active["split"]], 2, cv2.LINE_AA)

    tri = np.array([[cx, cy - 9], [cx - 6, cy + 6], [cx + 6, cy + 6]], np.int32)
    cv2.fillPoly(img, [tri], (255, 255, 255))
    cv2.polylines(img, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"{mpp * h:.0f} m", (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def active_at(lane, t, tol):
    for run in lane:
        if run["t"][0] - tol <= t <= run["t"][-1] + tol:
            i = int(np.argmin(np.abs(run["t"] - t)))
            if abs(run["t"][i] - t) <= tol:
                return run, i
    return None, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", default="zed2")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--step", type=float, default=0.5, help="Real seconds per output frame.")
    p.add_argument("--mpp", type=float, default=0.30, help="Map metres per pixel.")
    p.add_argument("--crf", type=int, default=30)
    p.add_argument("--poses", type=pathlib.Path,
                   default=DATALOCAL_ROOT / "wayfaster_split_videos" / "poses_opt.npz")
    p.add_argument("--out", type=pathlib.Path,
                   default=DATALOCAL_ROOT / "wayfaster_split_videos" / "nav_2023_06_28_zed2.mp4")
    args = p.parse_args()

    runs = [load_run(args.cam, s, f) for s, f in RUNS_2023_06_28]
    if args.poses and args.poses.exists():
        opt = np.load(args.poses)
        for r in runs:
            r["xy"], r["yaw"] = opt[r["name"]][:, :2], opt[r["name"]][:, 2]
        print(f"using corrected poses from {args.poses}")
    for r in runs:
        r["p0"], r["p1"] = r["xy"][0], r["xy"][-1]

    lanes = build_lanes(runs)
    for k, lane in enumerate(lanes):
        print(f"Lane {k}: " + " -> ".join(f"{r['name']}({r['split']})" for r in lane))

    h0, w0 = cv2.imread(runs[0]["paths"][0]).shape[:2]
    all_t = np.concatenate([r["t"] for r in runs])
    t0, t1 = all_t.min(), all_t.max()
    n = int(np.ceil((t1 - t0) / args.step)) + 1
    tol = args.step

    cw, ch = len(lanes) * w0, GLOBAL_BAR_H + BANNER_H + h0 + MAP_H
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(
        args.out, fps=args.fps, codec="libx264", macro_block_size=2,
        output_params=["-crf", str(args.crf), "-preset", "veryfast", "-pix_fmt", "yuv420p"],
    )

    for si in range(n):
        t = t0 + si * args.step
        canvas = np.zeros((ch, cw, 3), np.uint8)
        cv2.putText(canvas, f"t={t:.1f}", (6, GLOBAL_BAR_H - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for k, lane in enumerate(lanes):
            x0 = k * w0
            yb = GLOBAL_BAR_H
            run, i = active_at(lane, t, tol)
            canvas[yb : yb + BANNER_H, x0 : x0 + w0] = COLORS[run["split"]] if run else IDLE_COLOR
            label = f"LANE {k}  " + (f"{run['split'].upper()} {run['name']}" if run else "idle")
            cv2.putText(canvas, label, (x0 + 4, yb + BANNER_H - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)

            yi = yb + BANNER_H
            if run is not None:
                img = cv2.imread(run["paths"][i])
                if img is not None:
                    canvas[yi : yi + h0, x0 : x0 + w0] = img
                pose = (run["xy"][i, 0], run["xy"][i, 1], run["yaw"][i])
            else:
                pose = None

            ym = yi + h0
            canvas[ym : ym + MAP_H, x0 : x0 + w0] = (
                draw_map(runs, run, pose, w0, MAP_H, args.mpp)
                if pose is not None
                else np.full((MAP_H, w0, 3), 18, np.uint8)
            )
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"{args.out}: {n} frames @ {args.fps} fps, {len(lanes)} lanes")


if __name__ == "__main__":
    main()
