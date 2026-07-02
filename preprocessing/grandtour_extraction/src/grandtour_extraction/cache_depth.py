"""Cache full-res depth PNGs onto local SSD for the training dataloader.

Depth stays full resolution (needed for GPU voxelization, see
traversability_network's TravNet.voxelize), so unlike resize_images.py this
can't shrink the data by downsampling -- it re-encodes each 16-bit PNG at max
zlib compression (~19% smaller, bit-exact) into a local root that
traversability_network.config.DATASET.DEPTH_CACHE_ROOT points at.

Not part of pipeline.py: this is a machine-local training cache derived from
the canonical extracted/ output, not a step that mutates that output.
"""

import argparse
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from config.paths import DATALOCAL_ROOT, ensure_datalocal_root
from data_labeling_pipeline.pipeline_step import is_step_done, log_step, write_step_done
from tqdm import tqdm

VERSION = 2

_PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 9, cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_FILTERED]


def _recompress_one(src: pathlib.Path, dst: pathlib.Path) -> None:
    # Runs in a worker thread (cv2 encode/decode release the GIL) and possibly
    # a concurrent process (train.py's preflight runs per-DDP-rank,
    # unsynchronized). Write to a PID-unique temp file and atomically rename,
    # so two writers racing on the same dst can never produce a torn file.
    depth = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    _, buf = cv2.imencode(".png", depth, _PNG_PARAMS)
    tmp = dst.with_name(f"{dst.name}.tmp{os.getpid()}")
    tmp.write_bytes(buf)
    os.replace(tmp, dst)


def cache_mission(
    mission_dir: pathlib.Path,
    cache_root: pathlib.Path,
    n_workers: int,
    force: bool,
) -> None:
    if cache_root == DATALOCAL_ROOT or DATALOCAL_ROOT in cache_root.parents:
        ensure_datalocal_root()

    out_dir = cache_root / mission_dir.name
    if not force and is_step_done(out_dir, "cache_depth", VERSION):
        print(f"Skipping cache_depth (up to date): {out_dir}")
        return

    manifest = np.load(mission_dir / "depth_manifest.npz")
    rel_paths = [str(p) for p in manifest["depth"]["path"]]

    with log_step("cache_depth", out_dir):
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = []
            for rel in rel_paths:
                dst = out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                futures.append(pool.submit(_recompress_one, mission_dir / rel, dst))
            for f in tqdm(futures, desc=mission_dir.name):
                f.result()
        write_step_done(out_dir, "cache_depth", VERSION, count=len(rel_paths))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--extracted-root", required=True, type=pathlib.Path, help="DATASET.DATA_ROOT"
    )
    p.add_argument(
        "--cache-root",
        required=True,
        type=pathlib.Path,
        help="DATASET.DEPTH_CACHE_ROOT",
    )
    p.add_argument("--missions", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true", help="Re-run even if up to date.")
    args = p.parse_args()

    for mission in args.missions:
        cache_mission(
            args.extracted_root / mission, args.cache_root, args.workers, args.force
        )


if __name__ == "__main__":
    main()
