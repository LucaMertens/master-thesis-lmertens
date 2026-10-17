#!/usr/bin/env python
"""Clip of the deepest low-mu (RHE label <= --mu) incidents in the WayFASTER dataset."""

import argparse
import pathlib

import cv2
import imageio.v2 as iio
import numpy as np
import pandas as pd
from shared_utils.paths import WAYFASTER_DATASET_ROOT

FPS = 4.0  # 2x playback of 2 Hz images
CRF = 24


def parse_vec(text, n=2):
    v = np.fromstring(str(text).strip()[1:-1], sep=" ", dtype=np.float64)
    return np.pad(v, (0, max(0, n - v.size)), constant_values=np.nan)[:n]


def bag_frames(bag_dir):
    """Per-image (timestamp, path, mu), with mu taken from the most recent state."""
    states, images = bag_dir / "states.csv", bag_dir / "images.csv"
    if not (states.exists() and images.exists()):
        return None
    s, i = pd.read_csv(states), pd.read_csv(images)
    if "traversability" not in s or "image" not in i:
        return None
    mu = np.vstack(s["traversability"].map(parse_vec).to_numpy())[:, 0]
    ts_s, ts_i = s["timestamp"].to_numpy(float), i["timestamp"].to_numpy(float)
    idx = np.searchsorted(ts_s, ts_i, side="right") - 1
    ok = idx >= 0
    return ts_i[ok], i["image"].astype(str).to_numpy()[ok], mu[idx[ok]]


def incidents(ts, mu, thresh, pad_s, merge_s):
    """(start, end, argmin) index triples of the padded, merged low-mu stretches."""
    low = np.isfinite(mu) & (mu <= thresh)
    if not low.any():
        return []
    starts = np.flatnonzero(low & ~np.r_[False, low[:-1]])
    ends = np.flatnonzero(low & ~np.r_[low[1:], False])
    segs = []
    for a, b in zip(starts, ends):
        s = int(np.searchsorted(ts, ts[a] - pad_s, side="left"))
        e = int(np.searchsorted(ts, ts[b] + pad_s, side="right") - 1)
        segs.append((s, min(e, len(ts) - 1)))
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if ts[s] - ts[merged[-1][1]] <= merge_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e, s + int(np.nanargmin(mu[s : e + 1]))) for s, e in merged]


def trim(ts, s, e, peak, max_clip_s):
    """Window of at most max_clip_s around the mu minimum."""
    if max_clip_s is None or ts[e] - ts[s] <= max_clip_s:
        return s, e
    half = max_clip_s / 2
    return (int(np.searchsorted(ts, max(ts[s], ts[peak] - half), side="left")),
            int(np.searchsorted(ts, min(ts[e], ts[peak] + half), side="right") - 1))


def caption(img, text):
    """Draw a translucent label bar."""
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.34
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    x1, y1 = tw + 10, th + base + 8
    img[:y1, :x1] = cv2.addWeighted(img[:y1, :x1], 0.4, np.zeros_like(img[:y1, :x1]), 0.6, 0)
    cv2.putText(img, text, (5, 5 + th), font, scale, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=pathlib.Path, default=WAYFASTER_DATASET_ROOT)
    ap.add_argument("--out", default="wayfaster_incidents.mp4")
    ap.add_argument("--mu", type=float, default=0.5, help="incident threshold")
    ap.add_argument("--n", type=int, default=5, help="how many incidents to keep")
    ap.add_argument("--per-bag", type=int, default=1,
                    help="at most this many incidents from one recording")
    ap.add_argument("--max-clip-s", type=float, default=6.0)
    ap.add_argument("--pad-s", type=float, default=1.0)
    ap.add_argument("--merge-s", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--width", type=int, default=None,
                    help="upscale output frames to this width; source images are 320px native")
    args = ap.parse_args()

    found = []
    for states in sorted(args.root.rglob("states.csv")):
        rows = bag_frames(states.parent)
        if rows is None:
            continue
        ts, names, mu = rows
        for s, e, peak in incidents(ts, mu, args.mu, args.pad_s, args.merge_s):
            s, e = trim(ts, s, e, peak, args.max_clip_s)
            found.append((float(mu[peak]), states.parent, names[s : e + 1], mu[s : e + 1]))

    # deepest first, capped per recording
    found.sort(key=lambda f: (f[0], -len(f[2])))
    kept, per_bag = [], {}
    for f in found:
        if len(kept) >= args.n:
            break
        if per_bag.get(f[1], 0) < args.per_bag:
            kept.append(f)
            per_bag[f[1]] = per_bag.get(f[1], 0) + 1
    print(f"{len(found)} incidents below mu {args.mu}, keeping {len(kept)}")

    writer, n_frames = None, 0
    for peak_mu, bag_dir, names, mus in kept:
        for name, m in zip(names, mus):
            frame = cv2.imread(str(bag_dir / name))
            if frame is None:
                continue
            frame = caption(frame, f"{bag_dir.name[-13:]}   mu {m + 0.0:.2f}".replace("-0.00", "0.00"))
            if args.width:
                h = round(args.width * frame.shape[0] / frame.shape[1] / 2) * 2
                frame = cv2.resize(frame, (args.width, h), interpolation=cv2.INTER_CUBIC)
            if writer is None:
                h, w = frame.shape[:2]
                writer = iio.get_writer(
                    args.out, fps=args.fps, codec="libx264", macro_block_size=2,
                    output_params=["-crf", str(CRF), "-preset", "medium", "-pix_fmt", "yuv420p"])
            writer.append_data(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB))
            n_frames += 1
        print(f"  {bag_dir.name}: peak mu {peak_mu:.3f}, {len(names)} frames")
    if writer is not None:
        writer.close()
    print(f"{n_frames} frames, {n_frames / args.fps:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
