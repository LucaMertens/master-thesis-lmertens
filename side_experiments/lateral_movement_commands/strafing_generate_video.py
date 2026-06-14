import imageio_ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import zarr
from config.paths import DATASET_ROOT, OUTPUT_ROOT
from matplotlib.animation import FFMpegWriter, PillowWriter
from PIL import Image
from tqdm.auto import tqdm


def _seconds(t):
    t = np.asarray(t, dtype=float)
    return t / 1e9 if np.nanmax(np.abs(t)) > 1e12 else t


def _nearest_indices(reference_t, query_t):
    idx = np.searchsorted(reference_t, query_t)
    idx = np.clip(idx, 1, len(reference_t) - 1)
    left = idx - 1
    take_left = np.abs(query_t - reference_t[left]) <= np.abs(
        reference_t[idx] - query_t
    )
    return np.where(take_left, left, idx)


def _read_frame(path, max_width=960):
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.width > max_width:
            h = int(round(img.height * (max_width / img.width)))
            img = img.resize((max_width, h), Image.Resampling.BILINEAR)
        return np.asarray(img)


mission_dir = DATASET_ROOT / "huggingface" / "2024-10-01-11-47-44"
data_dir = mission_dir / "data"
images_dir = mission_dir / "images" / "zed2i_left_images"
output_dir = OUTPUT_ROOT / "side_experiments"

# Output controls
fps = 10
max_width = 960
max_frames = None  # Set int to limit frames; None keeps full sequence.
output_mode = "mp4"  # "mp4" (fast) or "gif" (slow)

if output_mode == "mp4":
    output_file = output_dir / f"{mission_dir.name}_zed2i_left_custom.mp4"
else:
    output_file = output_dir / f"{mission_dir.name}_zed2i_left_custom.gif"

# Use packaged ffmpeg binary when available (no system ffmpeg needed).
if output_mode == "mp4":
    try:
        mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        print(f"Could not resolve ffmpeg binary ({e}); falling back to GIF.")
        output_mode = "gif"
        output_file = output_dir / f"{mission_dir.name}_zed2i_left_custom.gif"

# Content controls
include_camera = True
include_stim_imu = False
include_anymal_imu = False
include_twist_cmd = True

if not (include_camera or include_stim_imu or include_anymal_imu or include_twist_cmd):
    raise ValueError("Enable at least one content stream.")

frame_paths_all = sorted(images_dir.glob("*.jpeg"))
if not frame_paths_all:
    raise FileNotFoundError(f"No frames found in {images_dir}")

cam_t_all = _seconds(
    zarr.open_array(data_dir / "zed2i_left_images" / "timestamp", mode="r")[:]
)
n_all = min(len(frame_paths_all), len(cam_t_all))
frame_paths_all = frame_paths_all[:n_all]
cam_t_all = cam_t_all[:n_all]

n_out = n_all if max_frames is None else min(n_all, int(max_frames))
sel = np.linspace(0, n_all - 1, num=n_out, dtype=int)
frame_paths = [frame_paths_all[i] for i in sel]
cam_t = cam_t_all[sel]

plot_items = []
if include_stim_imu:
    stim_t = _seconds(
        zarr.open_array(data_dir / "stim320_imu" / "timestamp", mode="r")[:]
    )
    stim_y = np.asarray(
        zarr.open_array(data_dir / "stim320_imu" / "lin_acc", mode="r")[:, 1],
        dtype=float,
    )
    plot_items.append({
        "name": "stim320_imu y",
        "t": stim_t,
        "y": stim_y,
        "idx": _nearest_indices(stim_t, cam_t),
    })

if include_anymal_imu:
    anymal_t = _seconds(
        zarr.open_array(data_dir / "anymal_imu" / "timestamp", mode="r")[:]
    )
    anymal_y = np.asarray(
        zarr.open_array(data_dir / "anymal_imu" / "lin_acc", mode="r")[:, 1],
        dtype=float,
    )
    plot_items.append({
        "name": "anymal_imu y",
        "t": anymal_t,
        "y": anymal_y,
        "idx": _nearest_indices(anymal_t, cam_t),
    })

if include_twist_cmd:
    twist_t = _seconds(
        zarr.open_array(data_dir / "anymal_command_twist" / "timestamp", mode="r")[:]
    )
    twist_y = np.asarray(
        zarr.open_array(data_dir / "anymal_command_twist" / "linear", mode="r")[:, 1],
        dtype=float,
    )
    plot_items.append({
        "name": "command_twist y",
        "t": twist_t,
        "y": twist_y,
        "idx": _nearest_indices(twist_t, cam_t),
    })

if include_camera and plot_items:
    fig, (ax_img, ax_sig) = plt.subplots(
        2, 1, figsize=(9, 5.8), dpi=90, gridspec_kw={"height_ratios": [3, 1]}
    )
elif include_camera:
    fig, ax_img = plt.subplots(1, 1, figsize=(9, 4.8), dpi=90)
    ax_sig = None
else:
    fig, ax_sig = plt.subplots(1, 1, figsize=(9, 3.6), dpi=90)
    ax_img = None

if include_camera:
    assert ax_img is not None
    img_artist = ax_img.imshow(_read_frame(frame_paths[0], max_width=max_width))
    title = ax_img.set_title("zed2i_left_images")
    ax_img.axis("off")

if plot_items:
    assert ax_sig is not None
    markers = []
    y_min = min(float(item["y"].min()) for item in plot_items)
    y_max = max(float(item["y"].max()) for item in plot_items)
    pad = 0.05 * (y_max - y_min + 1e-9)

    for item in plot_items:
        ax_sig.plot(item["t"], item["y"], lw=1.0, label=item["name"])
        (marker,) = ax_sig.plot([], [], "o", ms=4)
        markers.append(marker)

    time_line = ax_sig.axvline(cam_t[0], color="k", ls="--", lw=1)
    ax_sig.set_xlim(float(cam_t[0]), float(cam_t[-1]))
    ax_sig.set_ylim(y_min - pad, y_max + pad)
    ax_sig.set_xlabel("time [s]")
    ax_sig.set_ylabel("y-axis value")
    ax_sig.grid(alpha=0.3)
    ax_sig.legend(loc="upper right")

fig.tight_layout()


def _update(i):
    if include_camera:
        img_artist.set_data(_read_frame(frame_paths[i], max_width=max_width))
        title.set_text(f"zed2i_left_images | frame {i + 1}/{n_out}")

    if plot_items:
        t = cam_t[i]
        time_line.set_xdata([t, t])
        for item, marker in zip(plot_items, markers):
            j = item["idx"][i]
            marker.set_data([item["t"][j]], [item["y"][j]])


frame_iter = tqdm(range(n_out), desc=f"Encoding {output_mode.upper()}", unit="frame")

if output_mode == "mp4":
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"],
    )
    with writer.saving(fig, str(output_file), dpi=fig.dpi):
        for i in frame_iter:
            _update(i)
            writer.grab_frame()
else:
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(output_file), dpi=fig.dpi):
        for i in frame_iter:
            _update(i)
            writer.grab_frame()

plt.close(fig)
print(f"Saved {output_mode.upper()} with {n_out} frames @ {fps} FPS to: {output_file}")
