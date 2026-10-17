#!/usr/bin/env python
"""Render one video per WayFASTER timestamp folder, merging data_train and data_valid frames.

Output: <out-root>/<cam>/<ts_folder>.mp4
"""

import argparse
import pathlib

import cv2
import imageio.v2 as iio
import numpy as np
import pandas as pd
from shared_utils.paths import DATALOCAL_ROOT, WAYFASTER_DATASET_ROOT
from common import COLORS

CAMS = ["realsense", "zed2"]
SPLITS = ["train", "valid"]
SCALE = 1
BANNER_H = 28


def list_ts_folders(cam: str, root: pathlib.Path = WAYFASTER_DATASET_ROOT) -> list[str]:
    names = set()
    for split in SPLITS:
        d = root / cam / f"data_{split}"
        if d.is_dir():
            names.update(p.name for p in d.iterdir() if p.is_dir())
    return sorted(names)


def gather_frames(
    cam: str, ts_folder: str, root: pathlib.Path = WAYFASTER_DATASET_ROOT
) -> pd.DataFrame:
    """Frames from both splits for one ts folder, sorted by timestamp."""
    parts = []
    for split in SPLITS:
        csv_path = root / cam / f"data_{split}" / ts_folder / "images.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=["timestamp", "image"])
        df["path"] = str(csv_path.parent) + "/" + df["image"]
        df["split"] = split
        parts.append(df[["timestamp", "path", "split"]])
    if not parts:
        raise FileNotFoundError(f"No images.csv found for {cam}/{ts_folder}")
    frames = pd.concat(parts, ignore_index=True)
    return frames.sort_values(["timestamp", "split"], kind="stable").reset_index(drop=True)


def render_video(frames: pd.DataFrame, out_path: pathlib.Path, crf: int = 28) -> None:
    ts = frames["timestamp"].values
    dt = np.median(np.diff(ts)) if len(ts) > 1 else 1.0
    fps = float(1.0 / dt) if dt > 0 else 2.0

    h0, w0 = cv2.imread(frames["path"].iloc[0]).shape[:2]
    w, h = w0 * SCALE, h0 * SCALE

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        macro_block_size=2,
        output_params=["-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p"],
    )
    for row in frames.itertuples(index=False):
        img = cv2.imread(row.path)
        if img is None:
            continue
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((h + BANNER_H, w, 3), dtype=np.uint8)
        canvas[:BANNER_H] = COLORS[row.split]
        canvas[BANNER_H:] = img
        label = f"{row.split.upper()}  t={row.timestamp:.2f}"
        cv2.putText(
            canvas, label, (6, BANNER_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"{out_path}: {len(frames)} frames @ {fps:.2f} fps")


def make_video(cam: str, ts_folder: str, out_root: pathlib.Path, crf: int = 28) -> pathlib.Path:
    frames = gather_frames(cam, ts_folder)
    out_path = out_root / cam / f"{ts_folder}.mp4"
    render_video(frames, out_path, crf=crf)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ts_folder", nargs="?", help="Single timestamp folder to render.")
    p.add_argument("--cam", choices=CAMS, help="Camera; required with a single ts_folder.")
    p.add_argument("--all", action="store_true", help="Render every ts folder, both cams.")
    p.add_argument("--out-root", type=pathlib.Path, default=DATALOCAL_ROOT / "wayfaster_split_videos")
    p.add_argument("--crf", type=int, default=32)
    args = p.parse_args()

    if args.all:
        for cam in CAMS:
            for ts_folder in list_ts_folders(cam):
                make_video(cam, ts_folder, args.out_root, crf=args.crf)
        return

    if not args.ts_folder:
        p.error("Pass a ts_folder, or --all.")
    cam = args.cam
    if cam is None:
        matches = [c for c in CAMS if args.ts_folder in list_ts_folders(c)]
        if len(matches) != 1:
            p.error(f"--cam required; {args.ts_folder} found in {matches or 'neither camera'}")
        cam = matches[0]
    make_video(cam, args.ts_folder, args.out_root, crf=args.crf)


if __name__ == "__main__":
    main()
