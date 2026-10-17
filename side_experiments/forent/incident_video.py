"""Video of raw ForEnt frames inside the incident windows (see incidents.py).

    uv run python side_experiments/forent/incident_video.py --out incidents.mp4

--max-clip-s trims each span to its opening seconds, --per-type keeps only the most
severe spans of each entrapment kind.
"""

import argparse
import pathlib

import cv2
import imageio.v2 as iio
import numpy as np
from forent_extraction.extract import ENTRAPMENT_TYPES

from incidents import FORENT_LEAD_S
from paths import FORENT_EXTRACTED_ROOT

FPS = 15.0
CRF = 26


def sequences(root):
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("seq_"))


def spans(root, per_type=None):
    """(seq_dir, span) pairs; with per_type, keep the top spans per entrapment kind (severity, then length)."""
    found = [(d, s) for d in sequences(root)
             for s in (np.load(d / "entrapment.npy") if (d / "entrapment.npy").exists() else [])]
    if per_type is None:
        return found
    keep = set()
    for kind in range(len(ENTRAPMENT_TYPES)):
        of_kind = [i for i, (_, s) in enumerate(found) if int(s[2]) == kind]
        of_kind.sort(key=lambda i: (found[i][1][3], found[i][1][1] - found[i][1][0]), reverse=True)
        keep.update(of_kind[:per_type])
    return [ds for i, ds in enumerate(found) if i in keep]


def clip_frames(seq_dir, span, lead_s, max_clip_s=None):
    start, end, kind, severity = span
    if max_clip_s is not None:
        end = min(end, start + max_clip_s)
    manifest = np.load(seq_dir / "image_manifest.npz")["front_center"]
    ts = manifest["ts_ns"] * 1e-9
    sel = (ts >= start - lead_s) & (ts <= end)
    label = ENTRAPMENT_TYPES[int(kind)] if 0 <= kind < len(ENTRAPMENT_TYPES) else "unknown"
    for path, t in zip(manifest["path"][sel], ts[sel]):
        frame = cv2.imread(str(seq_dir / path))
        phase = "run-up" if t < start else "incident"
        text = f"{seq_dir.name}  {label}  sev {severity:.0f}  {phase}  t={t - start:+.1f}s"
        cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        yield frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(FORENT_EXTRACTED_ROOT))
    ap.add_argument("--out", default="forent_incidents.mp4")
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--lead-s", type=float, default=FORENT_LEAD_S)
    ap.add_argument("--max-clip-s", type=float, default=None,
                    help="trim each span to its first seconds")
    ap.add_argument("--per-type", type=int, default=None,
                    help="keep only this many spans of each entrapment kind")
    ap.add_argument("--width", type=int, default=None, help="downscale to this width")
    args = ap.parse_args()

    writer, n_clips, n_frames = None, 0, 0
    for seq_dir, span in spans(pathlib.Path(args.root), args.per_type):
        n_clips += 1
        for frame in clip_frames(seq_dir, span, args.lead_s, args.max_clip_s):
            if args.width and frame.shape[1] != args.width:
                h = round(args.width * frame.shape[0] / frame.shape[1] / 2) * 2
                frame = cv2.resize(frame, (args.width, h), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = iio.get_writer(
                    args.out, fps=args.fps, codec="libx264", macro_block_size=2,
                    output_params=["-crf", str(CRF), "-preset", "medium", "-pix_fmt", "yuv420p"])
            writer.append_data(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB))
            n_frames += 1
    if writer is not None:
        writer.close()
    print(f"{n_clips} incident clips, {n_frames} frames, "
          f"{n_frames / args.fps:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
