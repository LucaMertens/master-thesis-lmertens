#!/usr/bin/env python
"""Side-by-side video of revisit pairs at similar pose, within one run (intra) or across runs (inter)."""

import argparse
import itertools
import pathlib
import sys

import cv2
import imageio.v2 as iio
import numpy as np
from shared_utils.paths import DATALOCAL_ROOT
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_runs import RUNS_2023_06_28, load_run
from common import COLORS

BANNER_H = 26
FOOTER_H = 22
SCALE = 2


def _dyaw(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


def find_intra(run: dict, args):
    """Pairs within one run at similar pose but far apart in time."""
    xy, yaw, t = run["xy"], run["yaw"], run["t"]
    cands = []
    for i, j in cKDTree(xy).query_pairs(args.max_dist):
        if abs(t[i] - t[j]) < args.min_dt:
            continue
        da = _dyaw(yaw[i], yaw[j])
        if da > args.max_ang:
            continue
        cands.append((i, j, float(np.linalg.norm(xy[i] - xy[j])), float(da)))
    return _select(cands, xy, args)


def find_inter(a: dict, b: dict, args):
    """Pairs across two runs at similar pose in the shared frame."""
    cands = []
    for i, js in enumerate(cKDTree(a["xy"]).query_ball_tree(cKDTree(b["xy"]), args.max_dist)):
        for j in js:
            da = _dyaw(a["yaw"][i], b["yaw"][j])
            if da > args.max_ang:
                continue
            cands.append((i, j, float(np.linalg.norm(a["xy"][i] - b["xy"][j])), float(da)))
    return _select(cands, a["xy"], args)


def _select(cands, xy, args):
    """Closest pairs first, kept spatially spread so places are not repeated."""
    cands.sort(key=lambda c: c[2])
    chosen, places = [], []
    for i, j, d, da in cands:
        p = xy[i]
        if any(np.linalg.norm(p - q) < args.min_place_sep for q in places):
            continue
        chosen.append((i, j, d, da))
        places.append(p)
        if len(chosen) >= args.per_pair:
            break
    return chosen


def panel(run: dict, idx: int, tag: str, w: int, h: int) -> np.ndarray:
    img = cv2.resize(cv2.imread(run["paths"][idx]), (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((BANNER_H + h, w, 3), np.uint8)
    out[:BANNER_H] = COLORS[run["split"]]
    out[BANNER_H:] = img
    label = f"{tag}  {run['ts_folder'][-9:]}  {run['split'].upper()}  t={run['t'][idx]:.1f}"
    cv2.putText(out, label, (6, BANNER_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["inter", "intra"], default="inter")
    p.add_argument("--cam", default="zed2")
    p.add_argument("--max-dist", type=float, default=3.0, help="Max position gap (m).")
    p.add_argument("--max-ang", type=float, default=np.deg2rad(30), help="Max heading gap (rad).")
    p.add_argument("--min-dt", type=float, default=30.0, help="Min time separation, intra mode (s).")
    p.add_argument("--min-place-sep", type=float, default=12.0, help="Min spacing between chosen places (m).")
    p.add_argument("--per-pair", type=int, default=6, help="Max pairs per run or run-pair.")
    p.add_argument("--hold", type=int, default=4, help="Video frames held per pair.")
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--crf", type=int, default=30)
    p.add_argument("--desc", type=pathlib.Path, default=DATALOCAL_ROOT / "wayfaster_split_videos" / "desc.npz")
    p.add_argument("--poses", type=pathlib.Path, help="Pose-graph corrected poses to select pairs with.")
    p.add_argument("--out", type=pathlib.Path)
    args = p.parse_args()

    tag = "_opt" if args.poses else ""
    out = args.out or DATALOCAL_ROOT / "wayfaster_split_videos" / f"revisits_{args.mode}_run{tag}.mp4"
    runs = [load_run(args.cam, s, f) for s, f in RUNS_2023_06_28]
    if args.poses:
        opt = np.load(args.poses)
        for r in runs:
            r["xy"], r["yaw"] = opt[r["name"]][:, :2], opt[r["name"]][:, 2]
    desc = np.load(args.desc) if args.desc.exists() else None

    if args.mode == "intra":
        jobs = [(r, r, find_intra(r, args)) for r in runs]
    else:
        jobs = [(a, b, find_inter(a, b, args)) for a, b in itertools.combinations(runs, 2)]

    h0, w0 = cv2.imread(runs[0]["paths"][0]).shape[:2]
    w, h = w0 * SCALE, h0 * SCALE

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(
        out, fps=args.fps, codec="libx264", macro_block_size=2,
        output_params=["-crf", str(args.crf), "-preset", "veryfast", "-pix_fmt", "yuv420p"],
    )

    total, sims = 0, []
    for a, b, chosen in jobs:
        if not chosen:
            continue
        print(f"{a['name']}({a['split'][0]}) <-> {b['name']}({b['split'][0]}): {len(chosen)} pairs")
        for i, j, d, da in chosen:
            sim = ""
            if desc is not None:
                c = float(desc[a["name"]][i] @ desc[b["name"]][j])
                sims.append(c)
                sim = f"   DINO cos={c:.2f}"
            canvas = np.zeros((BANNER_H + h + FOOTER_H, 2 * w, 3), np.uint8)
            canvas[: BANNER_H + h, :w] = panel(a, i, "A", w, h)
            canvas[: BANNER_H + h, w:] = panel(b, j, "B", w, h)
            foot = f"dist={d:.2f} m   dheading={np.degrees(da):.1f} deg{sim}"
            cv2.putText(canvas, foot, (8, BANNER_H + h + FOOTER_H - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            for _ in range(args.hold):
                writer.append_data(rgb)
            total += 1
    writer.close()
    if sims:
        s = np.array(sims)
        print(f"\nDINO cosine over {len(s)} geometry-selected pairs: median {np.median(s):.2f}, {100*(s>0.7).mean():.0f}% above 0.7")
    print(f"{out}: {total} pairs, {total * args.hold} frames @ {args.fps} fps")


if __name__ == "__main__":
    main()
