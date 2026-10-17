"""Render front_center RGB (with depth overlay) next to RHE mu/nu and command graphs.

Output: OUTPUT_ROOT/label_videos/<mission>_v<version>.mp4 (bump LABEL_VIDEO_VERSION to re-render).
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
from shared_utils.paths import EXTRACTED_ROOT, LABELS_ROOT, OUTPUT_ROOT
from data_labeling_pipeline.pipeline_step import log_step

LABEL_VIDEO_VERSION = 2

CAMERA = "front_center"
PANEL_W = 900  # graph panel width in px
HEIGHT = 600  # frame height in px; source image is downscaled to this
DPI = 100

OBS_EPS = 1e-6  # observability at/below this counts as zero (curve hidden)
CMD_MAX_GAP = 1.0  # break command curves across gaps longer than this [s]
DEPTH_ALPHA = 0.35
DEPTH_MAX = 20.0  # colorbar range [m]


def _load_rhe(rhe_dir: pathlib.Path):
    t = np.load(rhe_dir / "rhe_timestamp.npy")
    labels = np.load(rhe_dir / "rhe_trav_labels.npy")  # mu, nu, delta_theta
    mu_obs = np.load(rhe_dir / "rhe_mu_observability.npy")
    nu_obs = np.load(rhe_dir / "rhe_nu_observability.npy")
    return t, labels[:, 0], labels[:, 1], mu_obs, nu_obs


def _load_command(extracted_dir: pathlib.Path):
    f = extracted_dir / "command.npy"
    if not f.exists():
        return None
    cmd = np.load(f)  # t, vx, vy, vz, wx, wy, wz
    return cmd[:, 0], cmd[:, 1], cmd[:, 6]


def _curves(extracted_dir: pathlib.Path, rhe_dir: pathlib.Path):
    """(t0, axes); each axis is (ylabel, legend, [curve dicts]) on one time origin."""
    t_rhe, mu, nu, mu_obs, nu_obs = _load_rhe(rhe_dir)
    t0 = t_rhe[0]
    tr = t_rhe - t0
    axes = [
        ("mu", False, [dict(label="mu", color="tab:blue", t=tr, y=mu, obs=mu_obs)]),
        ("nu", False, [dict(label="nu", color="tab:orange", t=tr, y=nu, obs=nu_obs)]),
    ]
    cmd = _load_command(extracted_dir)
    if cmd is not None:
        t_cmd, vx, wz = cmd
        tc = t_cmd - t0
        axes.append(
            (
                "command",
                True,
                [
                    dict(label="vx [m/s]", color="tab:green", t=tc, y=vx, gap=CMD_MAX_GAP),
                    dict(label="wz [rad/s]", color="tab:red", t=tc, y=wz, gap=CMD_MAX_GAP),
                ],
            )
        )
    return t0, axes


def _plot_xy(curve):
    """t, y for the line: hidden where obs is zero, broken across timestamp gaps."""
    t, y = curve["t"].copy(), curve["y"].copy()
    if curve.get("obs") is not None:
        y[curve["obs"] <= OBS_EPS] = np.nan
    gap = curve.get("gap")
    if gap is not None:
        breaks = np.where(np.diff(t) > gap)[0] + 1
        t = np.insert(t, breaks, np.nan)
        y = np.insert(y, breaks, np.nan)
    return t, y


def _render_panel(axes_spec, height: int):
    """Static graph image (BGR) plus per-curve pixel mappers for the cursor overlay."""
    n = len(axes_spec)
    fig, mpl_axes = plt.subplots(
        n, 1, sharex=True, figsize=(PANEL_W / DPI, height / DPI), dpi=DPI
    )
    if n == 1:
        mpl_axes = [mpl_axes]
    for ax, (ylabel, legend, curves) in zip(mpl_axes, axes_spec):
        for c in curves:
            t, y = _plot_xy(c)
            ax.plot(t, y, color=c["color"], lw=1.0, label=c["label"])
        ax.set_ylabel(ylabel, fontsize=8)
        if legend:
            ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    mpl_axes[-1].set_xlabel("t [s]", fontsize=8)
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    H = fig.canvas.get_width_height()[1]
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    panel = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR).copy()

    # Linear data->pixel maps, so per-frame drawing needs no matplotlib calls.
    xlim = mpl_axes[0].get_xlim()
    bb0 = mpl_axes[0].get_window_extent()
    x_slope = (bb0.x1 - bb0.x0) / (xlim[1] - xlim[0])
    x_int = bb0.x0 - xlim[0] * x_slope

    def x_at(t):
        return int(x_slope * t + x_int)

    y_top = int(H - mpl_axes[0].get_window_extent().y1)
    y_bot = int(H - mpl_axes[-1].get_window_extent().y0)

    rows = []  # cursor markers: py = slope * val + intercept, skipped where hidden
    for ax, (_, _, curves) in zip(mpl_axes, axes_spec):
        bb, ylim = ax.get_window_extent(), ax.get_ylim()
        ysl = (bb.y1 - bb.y0) / (ylim[1] - ylim[0])
        intercept = H - bb.y0 + ylim[0] * ysl
        text_x, text_bot = int(bb.x0) + 6, int(H - bb.y0) - 6
        for k, c in enumerate(curves):
            bgr = tuple(int(255 * v) for v in matplotlib.colors.to_rgb(c["color"]))[::-1]
            rows.append(
                dict(
                    label=c["label"],
                    bgr=bgr,
                    t=c["t"],
                    y=c["y"],
                    obs=c.get("obs"),
                    gap=c.get("gap"),
                    slope=-ysl,
                    intercept=intercept,
                    text_xy=(text_x, text_bot - 16 * (len(curves) - 1 - k)),
                )
            )

    plt.close(fig)
    return panel, x_at, y_top, y_bot, rows


def _visible(row, t) -> bool:
    """Whether this curve has a value to show at cursor time t."""
    if row["obs"] is not None and np.interp(t, row["t"], row["obs"]) <= OBS_EPS:
        return False
    if row["gap"] is not None:
        i = np.searchsorted(row["t"], t)
        i = min(max(i, 1), len(row["t"]) - 1)
        if min(abs(t - row["t"][i - 1]), abs(row["t"][i] - t)) > row["gap"]:
            return False
    return True


def _draw_frame(panel, x_at, y_top, y_bot, rows, t):
    frame = panel.copy()
    x = x_at(t)
    cv2.line(frame, (x, y_top), (x, y_bot), (60, 60, 60), 1)
    for row in rows:
        if not _visible(row, t):
            continue
        val = float(np.interp(t, row["t"], row["y"]))
        py = int(row["slope"] * val + row["intercept"])
        cv2.circle(frame, (x, py), 4, row["bgr"], -1)
        cv2.putText(
            frame,
            f"{row['label']}={val:+.3f}",
            row["text_xy"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            row["bgr"],
            1,
            cv2.LINE_AA,
        )
    return frame


def _overlay_depth(img_bgr, depth_path: pathlib.Path):
    """Alpha-blend a turbo-colorized depth map (mm->m) over the RGB frame."""
    dp = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if dp is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if dp.shape[:2] != (h, w):
        dp = cv2.resize(dp, (w, h), interpolation=cv2.INTER_NEAREST)
    depth_m = dp.astype(np.float32) / 1000.0
    valid = depth_m > 0
    norm = np.clip(depth_m / DEPTH_MAX, 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out = img_bgr.copy()
    out[valid] = cv2.addWeighted(img_bgr, 1 - DEPTH_ALPHA, color, DEPTH_ALPHA, 0)[valid]

    bar_h = int(0.4 * h)
    grad = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None].repeat(14, axis=1)
    bar = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
    y0 = 10
    out[y0 : y0 + bar_h, w - 24 : w - 10] = bar
    cv2.putText(out, f"{DEPTH_MAX:.0f}m", (w - 60, y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "0m", (w - 60, y0 + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_video(
    extracted_dir: pathlib.Path,
    rhe_dir: pathlib.Path,
    out_path: pathlib.Path,
    fps: float | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    depth: bool = True,
) -> None:
    img_dir = extracted_dir / "images" / CAMERA
    depth_dir = extracted_dir / "depth"
    all_pngs = sorted(img_dir.glob("*.png"))
    if not all_pngs:
        raise FileNotFoundError(f"No {CAMERA} images in {img_dir}")

    all_t = np.array([int(p.stem) for p in all_pngs], dtype=np.float64) * 1e-9
    if fps is None:  # base capture rate; stride then plays back faster than real time
        dt = np.median(np.diff(all_t))
        fps = float(1.0 / dt) if dt > 0 else 10.0

    pngs = all_pngs[::stride]
    t_img = all_t[::stride]
    if max_frames is not None:
        pngs, t_img = pngs[:max_frames], t_img[:max_frames]
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
    # disable=None: no bar when stdout is not a tty
    for i, (png, tr) in enumerate(
        tqdm(zip(pngs, t_rel), total=total, desc="label_video", disable=None)
    ):
        img = cv2.resize(cv2.imread(str(png)), (img_w, panel_h))
        if depth:
            dpath = depth_dir / png.name
            if dpath.exists():
                img = _overlay_depth(img, dpath)
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
    p.add_argument("--max-frames", type=int, default=None, help="Cap frame count.")
    p.add_argument("--no-depth", action="store_true", help="Disable depth overlay.")
    p.add_argument("--out", type=pathlib.Path, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.out is None and args.max_frames is None:
        extract(args.mission, fps=args.fps, stride=args.stride, force=args.force)
        return
    out = args.out or video_path(args.mission)
    make_video(
        EXTRACTED_ROOT / args.mission,
        LABELS_ROOT / args.mission / "rhe",
        out,
        fps=args.fps,
        stride=args.stride,
        max_frames=args.max_frames,
        depth=not args.no_depth,
    )


if __name__ == "__main__":
    main()
