"""Run the full per-mission preprocessing: extrinsics, images, RHE labels, depth.

uv run python preprocessing/pipeline.py                  # process everything not yet done
uv run python preprocessing/pipeline.py --status         # show per-mission step state
watch -n 10 uv run python preprocessing/pipeline.py --status   # live job view

Without --mission, all missions found under --bags-root are processed; steps that
are already up to date are skipped, so a bare invocation finishes whatever is
missing. With --parallel-missions > 1 (default 2), each mission logs to
<output>/<mission>/pipeline.log and the console only shows start/done/failed lines.
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from config.paths import OUTPUT_ROOT, ROS1_ROOT
from data_labeling_pipeline import labeling_pipeline, run_labels
from data_labeling_pipeline.pipeline_step import write_step_done
from grandtour_extraction import extract_alphasense, extract_dlio, rectify

sys.path.insert(0, str(pathlib.Path(__file__).parent / "depth_generation"))
import run_depth  # noqa: E402

DEFAULT_BAGS_ROOT = ROS1_ROOT
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT


def discover_missions(bags_root: pathlib.Path) -> list[str]:
    return sorted(
        d.name
        for d in pathlib.Path(bags_root).iterdir()
        if d.is_dir() and next(d.glob("*_alphasense.bag"), None) is not None
    )


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _depth_version() -> int:
    src = (
        pathlib.Path(__file__).parent / "depth_generation" / "generate_depth.py"
    ).read_text()
    return int(re.search(r"DEPTH_PIPELINE_VERSION = (\d+)", src).group(1))


# Files that prove a step ran before the metadata system existed.
STEP_OUTPUTS = {
    "extrinsics": ["cam2base.npy", "stereo_extrinsics.npz"],
    "images": ["image_manifest.npz", "intrinsics.npy"],
    "dlio": ["dlio_odom.npy"],
    "labels": ["trav_labels.npy", "timestamp.npy"],
    "depth": ["depth_manifest.npz"],
}


def adopt_existing(
    missions: list[str], output_root: pathlib.Path, steps: list[str] | None = None
) -> None:
    """Backfill *_metadata.json for outputs created before step versioning existed."""
    current = _current_versions()
    for mission in missions:
        mission_dir = pathlib.Path(output_root) / mission
        for step, outputs in STEP_OUTPUTS.items():
            if steps and step not in steps:
                continue
            meta = mission_dir / f"{step}_metadata.json"
            if meta.exists() or not all((mission_dir / f).exists() for f in outputs):
                continue
            write_step_done(mission_dir, step, current[step])
            print(f"{mission}: adopted {step} as v{current[step]}")


def _current_versions() -> dict[str, int]:
    return {
        "extrinsics": extract_alphasense.EXTRINSICS_VERSION,
        "images": extract_alphasense.IMAGES_VERSION,
        "dlio": extract_dlio.DLIO_VERSION,
        "labels": labeling_pipeline.LABELS_VERSION,
        "depth": _depth_version(),
    }


ACTIVE_SECONDS = 120


def _depth_running(mission_dir: pathlib.Path) -> str | None:
    """Progress string if depth generation is live for this mission.

    Primary signal is the depth_progress.json heartbeat written by the container.
    Fallback for runs without it: newest file mtime in depth/ (imwrite overwrites
    existing files on re-runs, so the directory mtime alone is not reliable).
    """
    progress = mission_dir / "depth_progress.json"
    if progress.exists() and time.time() - progress.stat().st_mtime < ACTIVE_SECONDS:
        p = json.loads(progress.read_text())
        return f"RUNNING {p['done']}/{p['total']}"
    depth_dir = mission_dir / "depth"
    if (mission_dir / "depth_metadata.json").exists() or not depth_dir.exists():
        return None
    files = list(depth_dir.iterdir())
    newest = max((f.stat().st_mtime for f in files), default=0)
    if time.time() - newest < ACTIVE_SECONDS:
        return f"RUNNING ({len(files)} written)"
    return None


def _running_step(mission_dir: pathlib.Path) -> str | None:
    """Currently executing extraction step, from the pipeline.log of this mission."""
    log_file = mission_dir / "pipeline.log"
    if not log_file.exists() or time.time() - log_file.stat().st_mtime > ACTIVE_SECONDS:
        return None
    open_steps: list[str] = []
    for line in log_file.read_text().splitlines():
        if m := re.search(r"Starting (\w+)", line):
            open_steps.append(m.group(1))
        elif (m := re.search(r"Finished (\w+)", line)) and m.group(1) in open_steps:
            open_steps.remove(m.group(1))
    return open_steps[-1] if open_steps else None


def print_status(missions: list[str], output_root: pathlib.Path) -> None:
    """Render the per-mission step state from the *_metadata.json files."""
    current = _current_versions()

    def cell(mission_dir: pathlib.Path, step: str, running: str | None) -> str:
        if step == running:
            return "RUNNING"
        if step == "depth" and (depth := _depth_running(mission_dir)):
            return depth
        meta = mission_dir / f"{step}_metadata.json"
        if not meta.exists():
            return "-"
        data = json.loads(meta.read_text())
        version = data.get("version")
        if version != current[step]:
            return f"stale (v{version})"
        return "no data" if data.get("no_data") else "done"

    def activity(mission_dir: pathlib.Path) -> str:
        mtimes = [
            f.stat().st_mtime
            for f in (mission_dir / "pipeline.log", mission_dir / "depth_progress.json")
            if f.exists()
        ]
        if not mtimes:
            return ""
        return time.strftime("%m-%d %H:%M", time.localtime(max(mtimes)))

    rows = []
    for m in missions:
        mission_dir = pathlib.Path(output_root) / m
        running = _running_step(mission_dir)
        rows.append([
            m,
            *(cell(mission_dir, s, running) for s in current),
            activity(mission_dir),
        ])
    header = ["mission", *current, "last activity"]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


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
    jobs: int | None = None,
    png_compression: int = 3,
    force: bool = False,
    log_to_file: bool = False,
) -> dict[str, str]:
    """Run the requested steps for one mission.

    Raises if images/extrinsics fail, since depth generation depends on their
    output. dlio/labels failures (e.g. a mission with no dlio bag) are
    non-fatal and returned as {step: error} so depth can still proceed.
    """
    mission_dir = pathlib.Path(bags_root) / mission
    output_dir = pathlib.Path(output_root) / mission
    if not mission_dir.exists():
        raise FileNotFoundError(mission_dir)

    if log_to_file:
        # One log file per mission and run; overwritten on re-runs so it cannot grow
        # unbounded. tqdm bars auto-disable because the file is not a tty.
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(output_dir / "pipeline.log", "w", buffering=1)
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())

    if images or extrinsics:
        log(f"== {mission}: alphasense ==")
        extract_alphasense.extract(
            mission_dir,
            output_dir,
            max_frames,
            balance,
            fov_scale,
            images,
            extrinsics,
            jobs,
            png_compression,
            force,
        )

    step_errors: dict[str, str] = {}
    if extrinsics:
        log(f"== {mission}: dlio ==")
        try:
            extract_dlio.extract(mission_dir, output_dir, force=force)
        except Exception as e:
            log(f"!! {mission}: dlio failed: {e}")
            step_errors["dlio"] = str(e)
    if labels:
        log(f"== {mission}: labels ==")
        try:
            run_labels(mission_dir, output_dir, force=force)
        except Exception as e:
            log(f"!! {mission}: labels failed: {e}")
            step_errors["labels"] = str(e)
    return step_errors


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mission",
        nargs="*",
        default=None,
        help="Mission(s) to process. Default: all missions found under --bags-root.",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help=(
            "Mission(s) to exclude when processing all missions. "
            "Only applied when --mission is not provided."
        ),
    )
    p.add_argument("--bags-root", type=pathlib.Path, default=DEFAULT_BAGS_ROOT)
    p.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--balance", type=float, default=rectify.DEFAULT_BALANCE)
    p.add_argument("--fov-scale", type=float, default=rectify.DEFAULT_FOV_SCALE)
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Worker threads for image extraction per mission "
        "(default: cpu_count // parallel-missions).",
    )
    p.add_argument(
        "--parallel-missions",
        type=int,
        default=2,
        help="Number of missions to process concurrently (separate processes).",
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
        "--no-depth",
        action="store_false",
        dest="depth",
        help="Skip depth generation (on by default; needs Docker+GPU host).",
    )
    p.add_argument("--gpus", default="all")
    p.add_argument(
        "--force", action="store_true", help="Re-run steps even if up to date."
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print per-mission step status and exit (combine with `watch` for a live view).",
    )
    p.add_argument(
        "--adopt-existing",
        nargs="*",
        default=None,
        choices=list(STEP_OUTPUTS),
        metavar="STEP",
        help="Mark steps whose output files exist but predate step versioning as done, then exit. "
        f"Optionally limit to specific steps: {', '.join(STEP_OUTPUTS)}. No steps = all.",
    )
    args = p.parse_args()

    if args.mission:
        missions = args.mission
    else:
        missions = discover_missions(args.bags_root)
        if args.exclude:
            excluded = set(args.exclude)
            unknown = [e for e in excluded if e not in missions]
            if unknown:
                print(f"Warning: excluded missions not found: {', '.join(unknown)}")
            missions = [m for m in missions if m not in excluded]

    if not missions:
        raise SystemExit(f"No missions to process under {args.bags_root}")

    if args.status:
        print_status(missions, args.output_root)
        return
    if args.adopt_existing is not None:
        adopt_existing(missions, args.output_root, args.adopt_existing)
        return

    jobs = args.jobs or max(1, (os.cpu_count() or 8) // args.parallel_missions)
    common_kwargs = dict(
        bags_root=args.bags_root,
        output_root=args.output_root,
        max_frames=args.max_frames,
        balance=args.balance,
        fov_scale=args.fov_scale,
        images=args.images,
        extrinsics=args.extrinsics,
        labels=args.labels,
        jobs=jobs,
        png_compression=args.png_compression,
        force=args.force,
    )

    failed: dict[str, str] = {}
    step_failed: dict[str, dict[str, str]] = {}
    if args.parallel_missions <= 1:
        for mission in missions:
            log(f"### {mission} ###")
            try:
                errors = run_pipeline(mission=mission, **common_kwargs)
                if errors:
                    step_failed[mission] = errors
            except Exception as e:
                log(f"!! {mission} failed: {e}")
                failed[mission] = str(e)
    else:
        log(
            f"{len(missions)} missions, {args.parallel_missions} in parallel, "
            f"{jobs} image-extraction threads each. "
            f"Per-mission logs: {args.output_root}/<mission>/pipeline.log"
        )
        with ProcessPoolExecutor(max_workers=args.parallel_missions) as pool:
            futures = {
                pool.submit(
                    run_pipeline, mission=mission, log_to_file=True, **common_kwargs
                ): mission
                for mission in missions
            }
            for fut in as_completed(futures):
                mission = futures[fut]
                try:
                    errors = fut.result()
                    if errors:
                        step_failed[mission] = errors
                    log(f"{mission}: done")
                except Exception as e:
                    log(f"!! {mission} failed: {e}")
                    failed[mission] = str(e)

    depth_failed: dict[str, str] = {}
    if args.depth:
        depth_missions = [m for m in missions if m not in failed]
        if depth_missions:
            log(f"== depth: {len(depth_missions)} mission(s) ==")
            try:
                depth_failed = run_depth.run_depth(
                    depth_missions, args.output_root, gpus=args.gpus, force=args.force
                )
            except Exception as e:
                log(f"!! depth failed: {e}")
                depth_failed = {m: str(e) for m in depth_missions}

    print("\n=== Report ===")
    n_ok = 0
    for mission in missions:
        notes = []
        if mission in failed:
            notes.append(f"FAILED - {failed[mission]}")
        for step, err in step_failed.get(mission, {}).items():
            notes.append(f"{step} FAILED - {err}")
        if mission in depth_failed:
            notes.append(f"depth FAILED - {depth_failed[mission]}")
        if notes:
            print(f"  {mission}: {', '.join(notes)}")
        else:
            print(f"  {mission}: OK")
            n_ok += 1
    print(f"{n_ok}/{len(missions)} mission(s) fully succeeded")

    if failed or depth_failed or step_failed:
        all_failed = sorted(set(failed) | set(depth_failed) | set(step_failed))
        raise SystemExit(f"Failed missions: {', '.join(all_failed)}")


if __name__ == "__main__":
    main()
