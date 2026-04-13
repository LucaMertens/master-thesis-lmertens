from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

TWIST_DIR = "anymal_command_twist"
IMU_DIR = "stim320_imu"


@dataclass
class StrafeMetrics:
    left_distance: float
    right_distance: float
    abs_distance: float
    samples: int


@dataclass
class MissionResult:
    mission: str
    twist: StrafeMetrics
    imu: StrafeMetrics


def _as_seconds(timestamps: np.ndarray) -> np.ndarray:
    """Convert common integer timestamp scales to seconds."""
    if timestamps.size == 0:
        return timestamps
    max_abs = np.nanmax(np.abs(timestamps))
    if max_abs > 1e14:
        return timestamps * 1e-9  # nanoseconds
    if max_abs > 1e11:
        return timestamps * 1e-6  # microseconds
    if max_abs > 1e8:
        return timestamps * 1e-3  # milliseconds
    return timestamps


def _integrate_signed(vy: np.ndarray, dt: np.ndarray) -> StrafeMetrics:
    left = float(np.sum(np.maximum(vy, 0.0) * dt))
    right = float(np.sum(np.maximum(-vy, 0.0) * dt))
    return StrafeMetrics(
        left_distance=left,
        right_distance=right,
        abs_distance=left + right,
        samples=int(vy.size),
    )


def _timestamps_to_dt(timestamps: np.ndarray, default_dt: float = 0.01) -> np.ndarray:
    if timestamps.size <= 1:
        return np.array([default_dt], dtype=float)
    ts_s = _as_seconds(timestamps.astype(float))
    dt = np.diff(ts_s, prepend=ts_s[0])
    dt[0] = np.median(dt[1:]) if dt.size > 1 else default_dt
    dt = np.nan_to_num(dt, nan=default_dt, posinf=default_dt, neginf=default_dt)
    dt[dt <= 0] = default_dt
    return dt


def _find_zarr_root_from_dir(source_dir: Path, required_array: str) -> Path | None:
    if (source_dir / ".zgroup").exists() and (source_dir / required_array).exists():
        return source_dir

    for zgroup in source_dir.rglob(".zgroup"):
        root = zgroup.parent
        if (root / required_array).exists():
            return root
    return None


def _load_signal_from_zarr_dir(
    source_dir: Path,
    zarr_array: str,
    zarr_column_idx: int,
    zarr_timestamp: str = "timestamp",
) -> tuple[np.ndarray, np.ndarray]:
    root = _find_zarr_root_from_dir(source_dir, zarr_array)
    if root is None:
        raise RuntimeError(f"No Zarr group with '{zarr_array}' found in {source_dir}")

    group = zarr.open_group(str(root), mode="r")
    if zarr_array not in group or zarr_timestamp not in group:
        raise RuntimeError(
            f"Missing required arrays in {root}: '{zarr_array}' and '{zarr_timestamp}'"
        )

    values = np.asarray(group[zarr_array], dtype=float)
    timestamps = np.asarray(group[zarr_timestamp], dtype=float).reshape(-1)
    if values.ndim == 1:
        signal = values.reshape(-1)
    else:
        if zarr_column_idx >= values.shape[1]:
            raise RuntimeError(
                f"Array '{zarr_array}' has shape {values.shape}; "
                f"cannot read column {zarr_column_idx}"
            )
        signal = values[:, zarr_column_idx].reshape(-1)

    n = min(signal.size, timestamps.size)
    return signal[:n], timestamps[:n]


def compute_twist_strafing(source_path: Path, default_dt: float) -> StrafeMetrics:
    vy, ts = _load_signal_from_zarr_dir(
        source_dir=source_path,
        zarr_array="linear",
        zarr_column_idx=1,
    )
    dt = _timestamps_to_dt(ts, default_dt=default_dt)
    n = min(vy.size, dt.size)
    return _integrate_signed(vy[:n], dt[:n])


def compute_imu_strafing(source_path: Path, default_dt: float) -> StrafeMetrics:
    ay, ts = _load_signal_from_zarr_dir(
        source_dir=source_path,
        zarr_array="lin_acc",
        zarr_column_idx=1,
    )
    dt = _timestamps_to_dt(ts, default_dt=default_dt)
    n = min(ay.size, dt.size)
    ay = ay[:n]
    dt = dt[:n]

    # Remove DC offset to reduce integration drift.
    ay = ay - np.nanmedian(ay)
    vy = np.cumsum(ay * dt)
    return _integrate_signed(vy, dt)


def _collect_topic_map(dataset_folder: Path, topic_dir_name: str) -> dict[str, Path]:
    topic_map: dict[str, Path] = {}
    for path in dataset_folder.rglob(topic_dir_name):
        if not path.is_dir():
            continue
        mission_root = path.parent.parent if path.parent.name == "data" else path.parent
        mission_key = str(mission_root.relative_to(dataset_folder))
        topic_map[mission_key] = path
    return topic_map


def collect_missions(dataset_folder: Path) -> list[tuple[str, Path, Path]]:
    twist_map = _collect_topic_map(dataset_folder, TWIST_DIR)
    imu_map = _collect_topic_map(dataset_folder, IMU_DIR)

    missions = sorted(set(twist_map).intersection(imu_map))
    return [(m, twist_map[m], imu_map[m]) for m in missions]


def visualize(results: list[MissionResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    missions = [r.mission for r in results]
    x = np.arange(len(results))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(results) * 0.6), 10), dpi=120)

    twist_left = [r.twist.left_distance for r in results]
    twist_right = [r.twist.right_distance for r in results]
    imu_left = [r.imu.left_distance for r in results]
    imu_right = [r.imu.right_distance for r in results]

    axes[0].bar(x - width / 2, twist_left, width=width, label="Twist Left")
    axes[0].bar(x + width / 2, twist_right, width=width, label="Twist Right")
    axes[0].set_title("Commanded Strafing Distance by Mission")
    axes[0].set_ylabel("Absolute distance [m-equivalent]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(missions, rotation=65, ha="right", fontsize=8)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x - width / 2, imu_left, width=width, label="IMU Left")
    axes[1].bar(x + width / 2, imu_right, width=width, label="IMU Right")
    axes[1].set_title("IMU-Derived Strafing Distance by Mission")
    axes[1].set_ylabel("Distance proxy [integrated m]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(missions, rotation=65, ha="right", fontsize=8)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "strafing_left_right_by_mission.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=120)
    twist_abs = [r.twist.abs_distance for r in results]
    imu_abs = [r.imu.abs_distance for r in results]
    ax2.scatter(twist_abs, imu_abs)
    for r, tx, im in zip(results, twist_abs, imu_abs):
        ax2.annotate(r.mission, (tx, im), fontsize=7, alpha=0.8)
    ax2.set_xlabel("Twist absolute strafing [m-equivalent]")
    ax2.set_ylabel("IMU absolute strafing proxy [integrated m]")
    ax2.set_title("Twist vs IMU Strafing per Mission")
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(output_dir / "strafing_twist_vs_imu_scatter.png")
    plt.close(fig2)


def save_summary_csv(results: list[MissionResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "strafing_summary.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mission",
                "twist_left",
                "twist_right",
                "twist_abs",
                "imu_left",
                "imu_right",
                "imu_abs",
                "twist_samples",
                "imu_samples",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.mission,
                    r.twist.left_distance,
                    r.twist.right_distance,
                    r.twist.abs_distance,
                    r.imu.left_distance,
                    r.imu.right_distance,
                    r.imu.abs_distance,
                    r.twist.samples,
                    r.imu.samples,
                ]
            )
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify left/right strafing from command twist and IMU for all missions."
    )
    parser.add_argument(
        "dataset_folder",
        type=Path,
        default=Path("/datahome/lmertens/grand_tour_dataset/"),
        help="Root dataset folder containing mission subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./strafing_analysis_outputs"),
        help="Directory where CSV summary and plots are stored.",
    )
    parser.add_argument(
        "--default-dt",
        type=float,
        default=0.01,
        help="Fallback sample period in seconds when no valid timestamps are found.",
    )
    args = parser.parse_args()

    dataset_folder = args.dataset_folder.expanduser().resolve()
    if not dataset_folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_folder}")

    missions = collect_missions(dataset_folder)
    if not missions:
        raise RuntimeError(
            "No mission pairs found. Expected both extracted topic folders "
            "'anymal_command_twist' and 'stim320_imu' per mission."
        )

    results: list[MissionResult] = []
    failures: list[tuple[str, str]] = []
    for mission, twist_source, imu_source in missions:
        try:
            twist_metrics = compute_twist_strafing(
                twist_source, default_dt=args.default_dt
            )
            imu_metrics = compute_imu_strafing(imu_source, default_dt=args.default_dt)
            results.append(
                MissionResult(mission=mission, twist=twist_metrics, imu=imu_metrics)
            )
            print(
                f"{mission}: twist_abs={twist_metrics.abs_distance:.3f}, "
                f"imu_abs={imu_metrics.abs_distance:.3f}"
            )
        except Exception as exc:
            failures.append((mission, str(exc)))

    if not results:
        raise RuntimeError(
            "Could not parse any mission successfully. First errors: "
            + "; ".join(f"{m}: {e}" for m, e in failures[:3])
        )

    csv_path = save_summary_csv(results, args.output_dir)
    visualize(results, args.output_dir)

    print(f"Saved summary: {csv_path}")
    print(
        f"Saved figures in: {args.output_dir.resolve()} "
        "(strafing_left_right_by_mission.png, strafing_twist_vs_imu_scatter.png)"
    )

    if failures:
        print("\nMissions that failed to parse:")
        for mission, err in failures:
            print(f"  - {mission}: {err}")


if __name__ == "__main__":
    main()
