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

from config.paths import EXTRACTED_ROOT, LABELS_ROOT, OUTPUT_ROOT, ROS1_ROOT
from data_labeling_pipeline import labeling_pipeline, run_labels
from grandtour_extraction import (
    correct_tint,
    extract_alphasense,
    extract_commands,
    extract_dlio,
    extract_feet,
    label_video,
    rectify,
)

sys.path.insert(0, str(pathlib.Path(__file__).parent / "depth_generation"))
import run_depth  # noqa: E402

DEFAULT_BAGS_ROOT = ROS1_ROOT
DEFAULT_OUTPUT_ROOT = EXTRACTED_ROOT
DEFAULT_LABELS_ROOT = LABELS_ROOT


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


def _current_versions() -> dict[str, int]:
    return {
        "extrinsics": extract_alphasense.EXTRINSICS_VERSION,
        "images": extract_alphasense.IMAGES_VERSION,
        "tint_correction": correct_tint.VERSION,
        "dlio": extract_dlio.DLIO_VERSION,
        "command": extract_commands.COMMAND_VERSION,
        "ground_z": extract_feet.GROUND_Z_VERSION,
        "rhe_labels": labeling_pipeline.LABELS_VERSION,
        "depth": _depth_version(),
        "label_video": label_video.LABEL_VIDEO_VERSION,
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


def _label_video_running(mission_dir: pathlib.Path) -> str | None:
    """Progress string if label_video rendering is live, from its heartbeat."""
    progress = mission_dir / "label_video_progress.json"
    if progress.exists() and time.time() - progress.stat().st_mtime < ACTIVE_SECONDS:
        p = json.loads(progress.read_text())
        return f"RUNNING {p['done']}/{p['total']}"
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


def print_status(
    missions: list[str], output_root: pathlib.Path, labels_root: pathlib.Path
) -> None:
    """Render the per-mission step state from the *_metadata.json files."""
    current = _current_versions()

    def cell(mission_dir: pathlib.Path, rhe_dir: pathlib.Path, step: str, running: str | None) -> str:
        if step == running:
            return "RUNNING"
        if step == "depth" and (depth := _depth_running(mission_dir)):
            return depth
        if step == "label_video":
            if lv := _label_video_running(mission_dir):
                return lv
            return label_video.video_status(mission_dir.name, OUTPUT_ROOT)
        meta_dir = rhe_dir if step == "rhe_labels" else mission_dir
        meta = meta_dir / f"{step}_metadata.json"
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
            for f in (
                mission_dir / "pipeline.log",
                mission_dir / "depth_progress.json",
                mission_dir / "label_video_progress.json",
            )
            if f.exists()
        ]
        if not mtimes:
            return ""
        return time.strftime("%m-%d %H:%M", time.localtime(max(mtimes)))

    rows = []
    for m in missions:
        mission_dir = pathlib.Path(output_root) / m
        rhe_dir = pathlib.Path(labels_root) / m / "rhe"
        running = _running_step(mission_dir)
        rows.append([
            m,
            *(cell(mission_dir, rhe_dir, s, running) for s in current),
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
    labels_root,
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
    output. dlio/command/labels failures (e.g. a mission with no dlio bag) are
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
    if images:
        log(f"== {mission}: tint_correction ==")
        try:
            correct_tint.correct(output_dir, threshold=1.5, local_window=5.0, force=force)
        except Exception as e:
            log(f"!! {mission}: tint_correction failed: {e}")
            step_errors["tint_correction"] = str(e)
    if extrinsics:
        log(f"== {mission}: dlio ==")
        try:
            extract_dlio.extract(mission_dir, output_dir, force=force)
        except Exception as e:
            log(f"!! {mission}: dlio failed: {e}")
            step_errors["dlio"] = str(e)
        log(f"== {mission}: command ==")
        try:
            extract_commands.extract(mission_dir, output_dir, force=force)
        except Exception as e:
            log(f"!! {mission}: command failed: {e}")
            step_errors["command"] = str(e)
        log(f"== {mission}: ground_z ==")
        try:
            extract_feet.extract(mission_dir, output_dir, force=force)
        except Exception as e:
            log(f"!! {mission}: ground_z failed: {e}")
            step_errors["ground_z"] = str(e)
    if labels:
        log(f"== {mission}: rhe_labels ==")
        try:
            labels_dir = pathlib.Path(labels_root) / mission / "rhe"
            run_labels(mission_dir, output_dir, labels_dir, force=force)
        except Exception as e:
            log(f"!! {mission}: rhe_labels failed: {e}")
            step_errors["rhe_labels"] = str(e)
    return step_errors


def _render_video(
    mission: str, output_root, labels_root, force: bool, log_to_file: bool
) -> None:
    """Worker for the parallel video stage: render one mission's label video."""
    if log_to_file:
        mission_dir = pathlib.Path(output_root) / mission
        mission_dir.mkdir(parents=True, exist_ok=True)
        # Append so the video stage does not clobber the extraction log.
        log_file = open(mission_dir / "pipeline.log", "a", buffering=1)
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())
    label_video.extract(mission, output_root, labels_root, OUTPUT_ROOT, force=force)


def run_videos(
    missions: list[str],
    output_root,
    labels_root,
    workers: int,
    force: bool = False,
) -> dict[str, str]:
    """Render label videos for all missions, workers missions in parallel."""
    failed: dict[str, str] = {}
    log_to_file = workers > 1
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _render_video, m, output_root, labels_root, force, log_to_file
            ): m
            for m in missions
        }
        for fut in as_completed(futures):
            mission = futures[fut]
            try:
                fut.result()
                log(f"{mission}: label_video done")
            except Exception as e:
                log(f"!! {mission}: label_video failed: {e}")
                failed[mission] = str(e)
    return failed


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
    p.add_argument("--labels-root", type=pathlib.Path, default=DEFAULT_LABELS_ROOT)
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
        "--videos",
        action="store_true",
        help="Render OUTPUT_ROOT/label_videos/<mission>.mp4 (off by default).",
    )
    p.add_argument(
        "--video-workers",
        type=int,
        default=None,
        help="Missions rendered in parallel for --videos "
        "(default: min(missions, cpu_count)).",
    )
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
        print_status(missions, args.output_root, args.labels_root)
        return

    jobs = args.jobs or max(1, (os.cpu_count() or 8) // args.parallel_missions)
    common_kwargs = dict(
        bags_root=args.bags_root,
        output_root=args.output_root,
        labels_root=args.labels_root,
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

    video_failed: dict[str, str] = {}
    if args.videos:
        video_missions = [m for m in missions if m not in failed]
        if video_missions:
            workers = args.video_workers or min(
                len(video_missions), os.cpu_count() or 8
            )
            log(
                f"== label_video: {len(video_missions)} mission(s), "
                f"{workers} in parallel =="
            )
            try:
                video_failed = run_videos(
                    video_missions,
                    args.output_root,
                    args.labels_root,
                    workers=workers,
                    force=args.force,
                )
            except Exception as e:
                log(f"!! label_video failed: {e}")
                video_failed = {m: str(e) for m in video_missions}

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
        if mission in video_failed:
            notes.append(f"label_video FAILED - {video_failed[mission]}")
        if notes:
            print(f"  {mission}: {', '.join(notes)}")
        else:
            print(f"  {mission}: OK")
            n_ok += 1
    print(f"{n_ok}/{len(missions)} mission(s) fully succeeded")

    if failed or depth_failed or step_failed or video_failed:
        all_failed = sorted(
            set(failed) | set(depth_failed) | set(step_failed) | set(video_failed)
        )
        raise SystemExit(f"Failed missions: {', '.join(all_failed)}")


if __name__ == "__main__":
    main()
