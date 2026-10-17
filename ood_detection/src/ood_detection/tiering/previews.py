"""Sped-up front_center clips, one per mission, for the tiering UI (video time = mission time / SPEED).

    uv run python -m ood_detection.tiering.previews [--force] [--missions ALB-1 ...]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import imageio.v2 as iio
import numpy as np

from shared_utils.paths import DATALOCAL_ROOT, LABELS_ROOT, OUTPUT_ROOT

RESIZED_ROOT = DATALOCAL_ROOT / "grand_tour_dataset_resized"
OUT_ROOT = OUTPUT_ROOT / "mission_tiering"

PREVIEW_DT = 1.0  # mission seconds between sampled frames
FPS = 20
SPEED = PREVIEW_DT * FPS  # playback speedup, and the video->mission time factor
WIDTH, HEIGHT = 256, 192  # 4:3, divisible by 16 for libx264
CRF = 30


def frame_times(mission):
    """(paths, absolute seconds) of the resized front_center frames, in time order."""
    d = RESIZED_ROOT / mission / "images/front_center"
    names = sorted(n for n in os.listdir(d) if n.endswith(".png"))
    ts = np.array([int(n[:-4]) for n in names], dtype=np.float64) * 1e-9
    return [str(d / n) for n in names], ts


def render(mission, force=False):
    """Writes <mission>.mp4 and returns its manifest entry."""
    out = OUT_ROOT / f"{mission}.mp4"
    label_t = np.load(LABELS_ROOT / mission / "rhe" / "rhe_timestamp.npy")
    t0, duration = float(label_t[0]), float(label_t[-1] - label_t[0])
    entry = {"mission": mission, "t0": t0, "duration": duration,
             "video": out.name, "speed": SPEED, "fps": FPS}
    if out.exists() and not force:
        return entry

    paths, ts = frame_times(mission)
    targets = t0 + np.arange(0, duration, PREVIEW_DT)
    idx = np.clip(np.searchsorted(ts, targets), 1, len(ts) - 1)
    idx = np.where(np.abs(ts[idx - 1] - targets) < np.abs(ts[idx] - targets), idx - 1, idx)

    tmp = out.with_suffix(".part.mp4")
    with iio.get_writer(tmp, fps=FPS, codec="libx264", quality=None, macro_block_size=None,
                        ffmpeg_params=["-crf", str(CRF), "-movflags", "+faststart"]) as w:
        for i in idx:
            im = cv2.imread(paths[i])
            w.append_data(cv2.cvtColor(cv2.resize(im, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2RGB))
    tmp.replace(out)
    return entry


def missions_on_disk():
    """Missions with resized frames and RHE labels."""
    return sorted(m for m in os.listdir(RESIZED_ROOT)
                  if (LABELS_ROOT / m / "rhe" / "rhe_timestamp.npy").exists()
                  and (RESIZED_ROOT / m / "images/front_center").is_dir())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--missions", nargs="*")
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    missions = args.missions or missions_on_disk()
    with ProcessPoolExecutor(args.jobs) as ex:
        entries = list(ex.map(render, missions, [args.force] * len(missions)))
    manifest = {"speed": SPEED, "fps": FPS, "width": WIDTH, "height": HEIGHT,
                "missions": {e["mission"]: e for e in entries}}
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"-> {OUT_ROOT} ({len(entries)} clips)")


if __name__ == "__main__":
    main()
