"""Render a per-mission video: front_center RGB next to RHE traversability and command graphs.

Graphs show the full timeline with a moving cursor; the value at the cursor is
printed on each axis. Output: OUTPUT_ROOT/label_videos/<mission>_v<version>.mp4
The version is in the filename so bumping LABEL_VIDEO_VERSION re-renders all missions.
"""

import argparse
import json
import pathlib

import cv2
import imageio.v2 as iio
import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config.paths import EXTRACTED_ROOT, LABELS_ROOT, OUTPUT_ROOT
from data_labeling_pipeline.pipeline_step import log_step

LABEL_VIDEO_VERSION = 1

CAMERA = "front_center"
PANEL_W = 900  # graph panel width in px
HEIGHT = 600  # frame height in px; source image is downscaled to this
DPI = 100


def _load_rhe(rhe_dir: pathlib.Path):
    t = np.load(rhe_dir / "rhe_timestamp.npy")
    labels = np.load(rhe_dir / "rhe_trav_labels.npy")  # mu, nu, delta_theta
    return t, labels[:, 0], labels[:, 1]


def _load_command(extracted_dir: pathlib.Path):
    f = extracted_dir / "command.npy"
    if not f.exists():
        return None
    cmd = np.load(f)  # t, vx, vy, vz, wx, wy, wz
    return cmd[:, 0], cmd[:, 1], cmd[:, 6]


def _curves(extracted_dir: pathlib.Path, rhe_dir: pathlib.Path):
    """Per-axis (label, color, t, y) curves, all on one time origin."""
    t_rhe, mu, nu = _load_rhe(rhe_dir)
    t0 = t_rhe[0]
    axes = [
        (
            "traversability",
            [("mu", "tab:blue", t_rhe - t0, mu), ("nu", "tab:orange", t_rhe - t0, nu)],
        ),
    ]
    cmd = _load_command(extracted_dir)
    if cmd is not None:
        t_cmd, vx, wz = cmd
        axes.append(("linear cmd [m/s]", [("vx", "tab:green", t_cmd - t0, vx)]))
        axes.append(("angular cmd [rad/s]", [("wz", "tab:red", t_cmd - t0, wz)]))
    return t0, axes


def _render_panel(axes_spec, height: int):
    """Static graph image (BGR) plus per-curve pixel mappers for the cursor overlay."""
    n = len(axes_spec)
    fig, mpl_axes = plt.subplots(
        n, 1, sharex=True, figsize=(PANEL_W / DPI, height / DPI), dpi=DPI
    )
    if n == 1:
        mpl_axes = [mpl_axes]
    for ax, (title, curves) in zip(mpl_axes, axes_spec):
        for label, color, t, y in curves:
            ax.plot(t, y, color=color, lw=1.0, label=label)
        ax.set_ylabel(title, fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    mpl_axes[-1].set_xlabel("t [s]", fontsize=8)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    H = fig.canvas.get_width_height()[1]
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    panel = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR).copy()

    # Precompute linear data->pixel maps so the per-frame path needs no matplotlib
    # calls (faster, and thread-safe for parallel frame rendering).
    xlim = mpl_axes[0].get_xlim()
    bb0 = mpl_axes[0].get_window_extent()
    x_slope = (bb0.x1 - bb0.x0) / (xlim[1] - xlim[0])
    x_int = bb0.x0 - xlim[0] * x_slope

    def x_at(t):
        return int(x_slope * t + x_int)

    y_top = int(H - mpl_axes[0].get_window_extent().y1)
    y_bot = int(H - mpl_axes[-1].get_window_extent().y0)

    rows = []  # (label, bgr, ts, ys, slope, intercept): py = slope * val + intercept
    for ax, (_, curves) in zip(mpl_axes, axes_spec):
        bb, ylim = ax.get_window_extent(), ax.get_ylim()
        ysl = (bb.y1 - bb.y0) / (ylim[1] - ylim[0])
        intercept = H - bb.y0 + ylim[0] * ysl
        for label, color, t, y in curves:
            bgr = tuple(int(255 * c) for c in matplotlib.colors.to_rgb(color))[::-1]
            rows.append((label, bgr, t, y, -ysl, intercept))

    plt.close(fig)
    return panel, x_at, y_top, y_bot, rows


def _draw_frame(panel, x_at, y_top, y_bot, rows, t):
    frame = panel.copy()
    x = x_at(t)
    cv2.line(frame, (x, y_top), (x, y_bot), (60, 60, 60), 1)
    for label, bgr, ts, ys, slope, intercept in rows:
        val = float(np.interp(t, ts, ys))
        py = int(slope * val + intercept)
        cv2.circle(frame, (x, py), 4, bgr, -1)
        cv2.putText(
            frame,
            f"{label}={val:+.3f}",
            (x + 6, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            bgr,
            1,
            cv2.LINE_AA,
        )
    return frame


def make_video(
    extracted_dir: pathlib.Path,
    rhe_dir: pathlib.Path,
    out_path: pathlib.Path,
    fps: float | None = None,
    stride: int = 1,
) -> None:
    img_dir = extracted_dir / "images" / CAMERA
    all_pngs = sorted(img_dir.glob("*.png"))
    if not all_pngs:
        raise FileNotFoundError(f"No {CAMERA} images in {img_dir}")

    all_t = np.array([int(p.stem) for p in all_pngs], dtype=np.float64) * 1e-9
    if fps is None:  # base capture rate; stride then plays back faster than real time
        dt = np.median(np.diff(all_t))
        fps = float(1.0 / dt) if dt > 0 else 10.0

    pngs = all_pngs[::stride]
    t_img = all_t[::stride]
    t0, axes_spec = _curves(extracted_dir, rhe_dir)
    t_rel = t_img - t0

    h0, w0 = cv2.imread(str(pngs[0])).shape[:2]
    panel, x_at, y_top, y_bot, rows = _render_panel(axes_spec, HEIGHT)
    panel_h = panel.shape[0]
    img_w = round(w0 * panel_h / h0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        macro_block_size=2,
        output_params=["-crf", "28", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
    )
    total = len(pngs)
    progress = extracted_dir / "label_video_progress.json"
    progress.write_text(json.dumps({"done": 0, "total": total}))
    # disable=None turns the bar off when stdout is not a tty (e.g. log files)
    for i, (png, tr) in enumerate(
        tqdm(zip(pngs, t_rel), total=total, desc="label_video", disable=None)
    ):
        img = cv2.resize(cv2.imread(str(png)), (img_w, panel_h))
        graph = _draw_frame(panel, x_at, y_top, y_bot, rows, tr)
        writer.append_data(cv2.cvtColor(cv2.hconcat([img, graph]), cv2.COLOR_BGR2RGB))
        if i % 50 == 0:
            progress.write_text(json.dumps({"done": i, "total": total}))
    writer.close()
    progress.unlink(missing_ok=True)
    print(f"label_video: {total} frames @ {fps:.1f} fps -> {out_path}")


def video_path(mission: str, output_root: pathlib.Path = OUTPUT_ROOT) -> pathlib.Path:
    return output_root / "label_videos" / f"{mission}_v{LABEL_VIDEO_VERSION}.mp4"


def _nodata_path(mission: str, output_root: pathlib.Path = OUTPUT_ROOT) -> pathlib.Path:
    return output_root / "label_videos" / f"{mission}_v{LABEL_VIDEO_VERSION}.nodata"


def video_status(mission: str, output_root: pathlib.Path = OUTPUT_ROOT) -> str:
    """Per-mission step state for --status, derived from the versioned filename."""
    out_dir = output_root / "label_videos"
    if video_path(mission, output_root).exists():
        return "done"
    if _nodata_path(mission, output_root).exists():
        return "no data"
    others = sorted(out_dir.glob(f"{mission}_v*.mp4"))
    if others:
        return f"stale (v{others[-1].stem.rsplit('_v', 1)[-1]})"
    return "-"


def extract(
    mission: str,
    extracted_root: pathlib.Path = EXTRACTED_ROOT,
    labels_root: pathlib.Path = LABELS_ROOT,
    output_root: pathlib.Path = OUTPUT_ROOT,
    fps: float | None = None,
    stride: int = 1,
    force: bool = False,
) -> None:
    extracted_dir = extracted_root / mission
    rhe_dir = labels_root / mission / "rhe"
    out_path = video_path(mission, output_root)

    if not force and (out_path.exists() or _nodata_path(mission, output_root).exists()):
        print(f"Skipping label_video (up to date): {out_path}")
        return

    with log_step("label_video", extracted_dir):
        if not (rhe_dir / "rhe_timestamp.npy").exists():
            print(f"No RHE labels in {rhe_dir}, skipping label_video")
            nodata = _nodata_path(mission, output_root)
            nodata.parent.mkdir(parents=True, exist_ok=True)
            nodata.touch()
            return
        make_video(extracted_dir, rhe_dir, out_path, fps=fps, stride=stride)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mission")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--stride", type=int, default=1, help="Use every Nth frame.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    extract(args.mission, fps=args.fps, stride=args.stride, force=args.force)


if __name__ == "__main__":
    main()
