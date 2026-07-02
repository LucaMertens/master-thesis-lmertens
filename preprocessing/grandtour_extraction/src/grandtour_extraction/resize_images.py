"""Pre-resize front_center color images for the training dataloader.

Training reads image_manifest.npz's native-res PNGs (~1440x1080) over NFS
only to immediately shrink them to the model's input size, which caps the
dataloader at a couple of iterations/s. This writes a same-relative-paths
copy at the target size into a local (non-NFS) root that
traversability_network.config.DATASET.RESIZED_ROOT points at.

Not part of pipeline.py: this is a machine-local training cache derived from
the canonical extracted/ output, not a step that mutates that output.
"""

import argparse
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from config.paths import DATALOCAL_ROOT, ensure_datalocal_root
from data_labeling_pipeline.pipeline_step import is_step_done, log_step, write_step_done
from tqdm import tqdm

VERSION = 1
CAM = "front_center"


def _resize_one(src: pathlib.Path, dst: pathlib.Path, size: tuple[int, int], png_compression: int) -> None:
    # Runs in a worker thread (cv2 releases the GIL) and possibly a concurrent
    # process (train.py's preflight runs per-DDP-rank, unsynchronized). Resize
    # to a PID-unique temp file and atomically rename, so two writers racing on
    # the same dst can never produce a torn/corrupt PNG.
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"failed to read {src}")
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    tmp = dst.with_name(f"tmp{os.getpid()}_{dst.name}")
    cv2.imwrite(str(tmp), img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    os.replace(tmp, dst)


def _up_to_date(out_dir: pathlib.Path, width: int, height: int) -> bool:
    if not is_step_done(out_dir, "resize_images", VERSION):
        return False
    meta = json.loads((out_dir / "resize_images_metadata.json").read_text())
    return meta.get("width") == width and meta.get("height") == height


def resize_mission(
    mission_dir: pathlib.Path,
    resized_root: pathlib.Path,
    width: int,
    height: int,
    n_workers: int,
    png_compression: int,
    force: bool,
) -> None:
    if resized_root == DATALOCAL_ROOT or DATALOCAL_ROOT in resized_root.parents:
        ensure_datalocal_root()

    out_dir = resized_root / mission_dir.name
    if not force and _up_to_date(out_dir, width, height):
        print(f"Skipping resize_images (up to date): {out_dir}")
        return

    manifest = np.load(mission_dir / "image_manifest.npz")
    rel_paths = [str(p) for p in manifest[CAM]["path"]]

    with log_step("resize_images", out_dir):
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = []
            for rel in rel_paths:
                dst = out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                futures.append(pool.submit(
                    _resize_one, mission_dir / rel, dst, (width, height), png_compression))
            for f in tqdm(futures, desc=mission_dir.name):
                f.result()
        write_step_done(out_dir, "resize_images", VERSION, count=len(rel_paths), width=width, height=height)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extracted-root", required=True, type=pathlib.Path, help="DATASET.DATA_ROOT")
    p.add_argument("--resized-root", required=True, type=pathlib.Path, help="DATASET.RESIZED_ROOT")
    p.add_argument("--missions", nargs="+", required=True)
    p.add_argument("--width", type=int, default=480, help="MODEL.INPUT_SIZE[0]")
    p.add_argument("--height", type=int, default=360, help="MODEL.INPUT_SIZE[1]")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--png-compression", type=int, default=1, help="0 (fast/large) - 9 (slow/small)")
    p.add_argument("--force", action="store_true", help="Re-run even if up to date.")
    args = p.parse_args()

    for mission in args.missions:
        resize_mission(
            args.extracted_root / mission, args.resized_root,
            args.width, args.height, args.workers, args.png_compression, args.force)


if __name__ == "__main__":
    main()
