"""Run the full per-mission preprocessing: extrinsics, images, RHE labels, (optional) depth.

uv run python -m grandtour_extraction.pipeline [--mission ETH-1 [ETH-2 ...]]
    [--no-images] [--no-extrinsics] [--no-labels] [--depth] [--max-frames 30]

Without --mission, all missions found under --bags-root are processed.
"""

import argparse
import pathlib

from data_labeling_pipeline import run_labels

from . import extract_extrinsics, extract_images, rectify, run_depth

DEFAULT_BAGS_ROOT = pathlib.Path("/datahome/lmertens/grand_tour_dataset/ros1")
DEFAULT_OUTPUT_ROOT = pathlib.Path("/datahome/lmertens/grand_tour_dataset/output")


def discover_missions(bags_root: pathlib.Path) -> list[str]:
    return sorted(
        d.name for d in pathlib.Path(bags_root).iterdir()
        if d.is_dir() and next(d.glob("*_alphasense.bag"), None) is not None
    )


def run_pipeline(
    mission: str,
    bags_root,
    output_root,
    max_frames,
    balance,
    fov_scale,
    images: bool,
    extrinsics: bool,
    labels: bool,
    depth: bool,
    gpus: str,
    jobs: int | None = None,
    png_compression: int = 3,
) -> None:
    mission_dir = pathlib.Path(bags_root) / mission
    output_dir = pathlib.Path(output_root) / mission
    if not mission_dir.exists():
        raise FileNotFoundError(mission_dir)

    if images:
        print(f"== {mission}: images ==")
        extract_images.extract(
            mission_dir, output_dir, max_frames, balance, fov_scale, jobs, png_compression,
        )
    if extrinsics:
        print(f"== {mission}: extrinsics ==")
        extract_extrinsics.extract(mission_dir, output_dir, balance, fov_scale)
    if labels:
        print(f"== {mission}: labels ==")
        run_labels(mission_dir, output_dir)
    if depth:
        print(f"== {mission}: depth ==")
        run_depth.run_depth(mission, output_root, gpus=gpus)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mission", nargs="*", default=None,
        help="Mission(s) to process. Default: all missions found under --bags-root.",
    )
    p.add_argument("--bags-root", type=pathlib.Path, default=DEFAULT_BAGS_ROOT)
    p.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--balance", type=float, default=rectify.DEFAULT_BALANCE)
    p.add_argument("--fov-scale", type=float, default=rectify.DEFAULT_FOV_SCALE)
    p.add_argument(
        "--jobs", type=int, default=None, help="Worker threads for image extraction."
    )
    p.add_argument(
        "--png-compression",
        type=int,
        default=3,
        choices=range(10),
        metavar="0-9",
        help="PNG compression level (0=fast/large, 9=slow/small).",
    )
    p.add_argument("--no-images", action="store_false", dest="images")
    p.add_argument("--no-extrinsics", action="store_false", dest="extrinsics")
    p.add_argument("--no-labels", action="store_false", dest="labels")
    p.add_argument(
        "--depth", action="store_true", help="Also run depth (needs Docker+GPU host)."
    )
    p.add_argument("--gpus", default="all")
    args = p.parse_args()

    missions = args.mission if args.mission else discover_missions(args.bags_root)
    if not missions:
        raise SystemExit(f"No missions found under {args.bags_root}")

    failed = []
    for mission in missions:
        print(f"### {mission} ###")
        try:
            run_pipeline(
                mission=mission,
                bags_root=args.bags_root,
                output_root=args.output_root,
                max_frames=args.max_frames,
                balance=args.balance,
                fov_scale=args.fov_scale,
                images=args.images,
                extrinsics=args.extrinsics,
                labels=args.labels,
                depth=args.depth,
                gpus=args.gpus,
                jobs=args.jobs,
                png_compression=args.png_compression,
            )
        except Exception as e:
            print(f"!! {mission} failed: {e}")
            failed.append(mission)

    if failed:
        raise SystemExit(f"Failed missions: {', '.join(failed)}")


if __name__ == "__main__":
    main()
