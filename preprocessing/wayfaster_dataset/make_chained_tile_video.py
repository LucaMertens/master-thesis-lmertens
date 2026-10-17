#!/usr/bin/env python
"""Render WayFASTER runs as lanes, chaining a run into a lane when it starts where the previous one ended."""

import argparse
import pathlib

import cv2
import imageio.v2 as iio
import numpy as np
import pandas as pd
from shared_utils.paths import DATALOCAL_ROOT, WAYFASTER_DATASET_ROOT
from scipy.optimize import linear_sum_assignment
from common import COLORS, RUNS_2023_06_28, nearest_frame, parse_vec

IDLE_COLOR = (40, 40, 40)
BANNER_H = 20
GLOBAL_BAR_H = 24
CLUSTER_GAP_S = 300.0  # start-time gap between run waves
CHAIN_DIST_THRESH = 30.0  # max end->start distance (m)


def load_run(cam: str, split: str, ts_folder: str, root: pathlib.Path = WAYFASTER_DATASET_ROOT) -> dict:
    run_dir = root / cam / f"data_{split}" / ts_folder
    images = pd.read_csv(run_dir / "images.csv", usecols=["timestamp", "image"]).sort_values("timestamp")
    states = pd.read_csv(run_dir / "states.csv", usecols=["timestamp", "position"])
    return dict(
        ts_folder=ts_folder,
        split=split,
        t=images["timestamp"].to_numpy(),
        paths=(str(run_dir) + "/" + images["image"]).to_numpy(),
        p0=parse_vec(states["position"].iloc[0])[:2],
        p1=parse_vec(states["position"].iloc[-1])[:2],
    )


def build_lanes(runs: list[dict]) -> list[list[dict]]:
    """Chain runs whose end/start positions are within CHAIN_DIST_THRESH into lanes."""
    runs = sorted(runs, key=lambda r: r["t"][0])
    clusters, cur = [], [runs[0]]
    for r in runs[1:]:
        if r["t"][0] - cur[-1]["t"][0] > CLUSTER_GAP_S:
            clusters.append(cur)
            cur = [r]
        else:
            cur.append(r)
    clusters.append(cur)

    lanes = [[r] for r in clusters[0]]
    for cluster in clusters[1:]:
        cost = np.array([[np.linalg.norm(lane[-1]["p1"] - r["p0"]) for r in cluster] for lane in lanes])
        row, col = linear_sum_assignment(cost)
        matched_runs, matched_lanes = set(), set()
        for i, j in zip(row, col):
            if cost[i, j] < CHAIN_DIST_THRESH:
                lanes[i].append(cluster[j])
                matched_runs.add(j)
                matched_lanes.add(i)
        for j, r in enumerate(cluster):
            if j in matched_runs:
                continue
            free = [k for k in range(len(lanes)) if k not in matched_lanes]
            if free:
                lanes[free[0]].append(r)
                matched_lanes.add(free[0])
            else:
                lanes.append([r])
    return lanes


def active_run(lane: list[dict], t: float, tol: float):
    for run in lane:
        if run["t"][0] - tol <= t <= run["t"][-1] + tol:
            return run
    return None


def render(lanes: list[list[dict]], out_path: pathlib.Path, fps: float, step: float, crf: int) -> None:
    h0, w0 = cv2.imread(lanes[0][0]["paths"][0]).shape[:2]
    all_t = np.concatenate([r["t"] for lane in lanes for r in lane])
    t_start, t_end = all_t.min(), all_t.max()
    n_steps = int(np.ceil((t_end - t_start) / step)) + 1
    tol = step

    canvas_w = len(lanes) * w0
    canvas_h = GLOBAL_BAR_H + h0 + BANNER_H

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        macro_block_size=2,
        output_params=["-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p"],
    )
    for step_i in range(n_steps):
        t = t_start + step_i * step
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        cv2.putText(
            canvas, f"t={t:.2f}", (6, GLOBAL_BAR_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        for k, lane in enumerate(lanes):
            x0, y0 = k * w0, GLOBAL_BAR_H
            run = active_run(lane, t, tol)
            banner_color = COLORS[run["split"]] if run else IDLE_COLOR
            label = f"LANE {k}  " + (f"{run['split'].upper()} {run['ts_folder'][-9:]}" if run else "idle")
            canvas[y0 : y0 + BANNER_H, x0 : x0 + w0] = banner_color
            cv2.putText(
                canvas, label, (x0 + 4, y0 + BANNER_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            if run is not None:
                path = nearest_frame(run, t, tol)
                if path is not None:
                    img = cv2.imread(path)
                    if img is not None:
                        canvas[y0 + BANNER_H : y0 + BANNER_H + h0, x0 : x0 + w0] = img
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"{out_path}: {n_steps} frames @ {fps:.1f} fps, {len(lanes)} lanes")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", default="zed2")
    p.add_argument("--fps", type=float, default=10.0, help="Encoder fps (playback speed).")
    p.add_argument("--step", type=float, default=0.5, help="Real seconds advanced per frame.")
    p.add_argument("--crf", type=int, default=32)
    p.add_argument(
        "--out", type=pathlib.Path,
        default=DATALOCAL_ROOT / "wayfaster_split_videos" / "tiles" / "2023_06_28_zed2_chained.mp4",
    )
    args = p.parse_args()

    runs = [load_run(args.cam, split, ts_folder) for split, ts_folder in RUNS_2023_06_28]
    lanes = build_lanes(runs)
    for k, lane in enumerate(lanes):
        seq = " -> ".join(f"{r['ts_folder']}({r['split']})" for r in lane)
        print(f"Lane {k}: {seq}")
    render(lanes, args.out, fps=args.fps, step=args.step, crf=args.crf)


if __name__ == "__main__":
    main()
