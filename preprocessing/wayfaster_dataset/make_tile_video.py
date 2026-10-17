#!/usr/bin/env python
"""Render several WayFASTER runs as a time-aligned tile grid with TRAIN/VALID banners."""

import argparse
import math
import pathlib

import cv2
import imageio.v2 as iio
import numpy as np
import pandas as pd
from shared_utils.paths import DATALOCAL_ROOT, WAYFASTER_DATASET_ROOT
from common import COLORS, RUNS_2023_06_28, nearest_frame

BANNER_H = 20
GLOBAL_BAR_H = 24


def load_run(cam: str, split: str, ts_folder: str, root: pathlib.Path = WAYFASTER_DATASET_ROOT):
    csv_path = root / cam / f"data_{split}" / ts_folder / "images.csv"
    df = pd.read_csv(csv_path, usecols=["timestamp", "image"])
    df["path"] = str(csv_path.parent) + "/" + df["image"]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return dict(
        ts_folder=ts_folder,
        split=split,
        t=df["timestamp"].to_numpy(),
        paths=df["path"].to_numpy(),
    )


def render_tile_video(
    runs: list[dict], out_path: pathlib.Path, ncols: int, fps: float, crf: int
) -> None:
    h0, w0 = cv2.imread(runs[0]["paths"][0]).shape[:2]
    nrows = math.ceil(len(runs) / ncols)
    dt = 1.0 / fps

    all_t = np.concatenate([r["t"] for r in runs])
    t_start, t_end = all_t.min(), all_t.max()
    n_steps = int(math.ceil((t_end - t_start) / dt)) + 1

    canvas_w = ncols * w0
    canvas_h = nrows * (h0 + BANNER_H) + GLOBAL_BAR_H

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        macro_block_size=2,
        output_params=["-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p"],
    )
    tol = dt

    for step in range(n_steps):
        t = t_start + step * dt
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        cv2.putText(
            canvas, f"t={t:.2f}", (6, GLOBAL_BAR_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        for k, run in enumerate(runs):
            row, col = divmod(k, ncols)
            x0 = col * w0
            y0 = GLOBAL_BAR_H + row * (h0 + BANNER_H)

            canvas[y0 : y0 + BANNER_H, x0 : x0 + w0] = COLORS[run["split"]]
            label = f"{run['split'].upper()} {run['ts_folder'][-9:]}"
            cv2.putText(
                canvas, label, (x0 + 4, y0 + BANNER_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

            path = nearest_frame(run, t, tol)
            if path is not None:
                img = cv2.imread(path)
                if img is not None:
                    canvas[y0 + BANNER_H : y0 + BANNER_H + h0, x0 : x0 + w0] = img
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"{out_path}: {n_steps} frames @ {fps:.2f} fps, {nrows}x{ncols} grid")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", default="zed2")
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--crf", type=int, default=32)
    p.add_argument(
        "--out", type=pathlib.Path,
        default=DATALOCAL_ROOT / "wayfaster_split_videos" / "tiles" / "2023_06_28_zed2.mp4",
    )
    args = p.parse_args()

    runs = [load_run(args.cam, split, ts_folder) for split, ts_folder in RUNS_2023_06_28]
    render_tile_video(runs, args.out, ncols=args.ncols, fps=args.fps, crf=args.crf)


if __name__ == "__main__":
    main()
