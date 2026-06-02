"""Preprocess one mission: rectified images -> extrinsics -> (optional) depth.

    uv run python -m grandtour_extraction.preprocess --mission ETH-1 [--max-frames 30] [--jobs 8] [--depth]
"""

import argparse
import pathlib

from . import extract_extrinsics, extract_images, rectify, run_depth

DEFAULT_BAGS_ROOT = pathlib.Path("/datahome/lmertens/grand_tour_dataset/ros1")
DEFAULT_OUTPUT_ROOT = pathlib.Path("/datahome/lmertens/grand_tour_dataset/output")


def preprocess(mission: str, bags_root, output_root, max_frames, balance, fov_scale,
               depth: bool, gpus: str, jobs: int | None = None,
               png_compression: int = 3) -> None:
    mission_dir = pathlib.Path(bags_root) / mission
    output_dir = pathlib.Path(output_root) / mission
    if not mission_dir.exists():
        raise FileNotFoundError(mission_dir)

    print(f"== {mission}: images ==")
    extract_images.extract(mission_dir, output_dir, max_frames, balance, fov_scale,
                           jobs, png_compression)
    print(f"== {mission}: extrinsics ==")
    extract_extrinsics.extract(mission_dir, output_dir, balance, fov_scale)
    if depth:
        print(f"== {mission}: depth ==")
        run_depth.run_depth(mission, output_root, gpus=gpus)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mission", required=True)
    p.add_argument("--bags-root", type=pathlib.Path, default=DEFAULT_BAGS_ROOT)
    p.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--balance", type=float, default=rectify.DEFAULT_BALANCE)
    p.add_argument("--fov-scale", type=float, default=rectify.DEFAULT_FOV_SCALE)
    p.add_argument("--jobs", type=int, default=None, help="Worker threads for image extraction.")
    p.add_argument("--png-compression", type=int, default=3, choices=range(10),
                   metavar="0-9", help="PNG compression level (0=fast/large, 9=slow/small).")
    p.add_argument("--depth", action="store_true", help="Also run depth (needs Docker+GPU host).")
    p.add_argument("--gpus", default="all")
    args = p.parse_args()
    preprocess(args.mission, args.bags_root, args.output_root, args.max_frames,
               args.balance, args.fov_scale, args.depth, args.gpus, args.jobs,
               args.png_compression)


if __name__ == "__main__":
    main()
