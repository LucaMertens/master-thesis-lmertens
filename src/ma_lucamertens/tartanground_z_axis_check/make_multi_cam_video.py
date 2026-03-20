# Create a video from the pngs in data/WaterMillDay/P1000/image_lcam_bottom.
# Next to it, display the corresponding z-axis imu data and HDIF cost, as previously calculated.

# Allows multiple camera folders to be rendered into a single video.

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from scipy.signal import welch

# ------------------------------
# Config
# ------------------------------
# dataset_root = Path("data") / "WaterMillDay" / "P1000"
dataset_root = Path("data") / "ForestEnv" / "Data_diff" / "P1000"
imu_dir = dataset_root / "imu"
camera_folders = ["image_lcam_bottom", "image_lcam_front"]
output_video = "imu_overlay_multi_cam.mp4"
img_fps = 10
imu_fps = 100
window_sec = 1.0
f_min, f_max = 1.0, 30.0

# ------------------------------
# Load IMU data
# ------------------------------
acc_path = imu_dir / "acc_nograv_body.npy"
imu_time_path = imu_dir / "imu_time.npy"
cam_time_path = imu_dir / "cam_time.npy"

acc_nograv_body = np.load(acc_path)
imu_time = np.load(imu_time_path)
cam_time = np.load(cam_time_path)

# Use z-axis acceleration
z_acc = acc_nograv_body[:, 2]


# ------------------------------
# HDIF cost calculation
# ------------------------------
def bandpower(data, sf, band, window_sec=None):
    band = np.asarray(band)
    low, high = band
    nperseg = int(window_sec * sf) if window_sec is not None else None
    freqs, psd = welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return simpson(psd[idx_band], dx=freq_res)


def rolling_cost(acc_z, timestamps, sf):
    window_samples = int(window_sec * sf)
    costs = np.full_like(acc_z, fill_value=np.nan, dtype=float)
    for i in range(window_samples, len(acc_z)):
        window = acc_z[i - window_samples : i]
        upper_freq = min(f_max, sf // 2 - 1)
        costs[i] = bandpower(window, sf, [f_min, upper_freq], window_sec=window_sec)
    return costs


costs = rolling_cost(z_acc, imu_time, imu_fps)

# Map each image timestamp to nearest imu index
imu_indices = np.searchsorted(imu_time, cam_time, side="left")
imu_indices = np.clip(imu_indices, 0, len(imu_time) - 1)

# ------------------------------
# Load camera images
# ------------------------------
camera_images = {}
for camera_folder in camera_folders:
    img_dir = dataset_root / camera_folder
    image_paths = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    camera_images[camera_folder] = image_paths

min_frame_count = min(len(cam_time), *(len(v) for v in camera_images.values()))
if min_frame_count == 0:
    raise ValueError("No frames available to render.")

# ------------------------------
# Build video with side plots
# ------------------------------
num_cams = len(camera_folders)
img_cols = 2 if num_cams > 1 else 1
img_rows = int(np.ceil(num_cams / img_cols))
fig_height = max(5, 3 * img_rows)
fig = plt.figure(figsize=(12, fig_height))
gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1.2], height_ratios=[1, 1])
img_gs = gs[:, 0].subgridspec(img_rows, img_cols)

ax_imgs = []
for idx in range(img_rows * img_cols):
    ax = fig.add_subplot(img_gs[idx // img_cols, idx % img_cols])
    ax.axis("off")
    ax_imgs.append(ax)

ax_z = fig.add_subplot(gs[0, 1])
ax_cost = fig.add_subplot(gs[1, 1])

# Pre-plot time series for context
ax_z.plot(imu_time, z_acc, color="tab:blue", linewidth=0.8)
ax_z.set_title("Z-axis IMU (acc_nograv_body)")
ax_z.set_ylabel("Accel")
ax_z.set_xlim(imu_time[0], imu_time[-1])
ax_z.grid(True, alpha=0.3)

ax_cost.plot(imu_time, costs, color="tab:red", linewidth=0.8)
ax_cost.set_title("HDIF Cost (1-30 Hz)")
ax_cost.set_xlabel("Time (s)")
ax_cost.set_ylabel("Cost")
ax_cost.set_xlim(imu_time[0], imu_time[-1])
ax_cost.grid(True, alpha=0.3)

# Moving markers
z_marker = ax_z.axvline(imu_time[0], color="black", linestyle="--", linewidth=1)
cost_marker = ax_cost.axvline(imu_time[0], color="black", linestyle="--", linewidth=1)

with imageio.get_writer(output_video, fps=img_fps) as writer:
    for frame_idx in range(min_frame_count):
        for cam_idx, camera_folder in enumerate(camera_folders):
            ax = ax_imgs[cam_idx]
            img_path = camera_images[camera_folder][frame_idx]
            img = imageio.imread(img_path)
            ax.imshow(img)
            ax.set_title(camera_folder)
            ax.axis("off")
        for ax in ax_imgs[num_cams:]:
            ax.cla()
            ax.axis("off")

        imu_idx = imu_indices[frame_idx]
        current_t = imu_time[imu_idx]
        z_marker.set_xdata([current_t, current_t])
        cost_marker.set_xdata([current_t, current_t])

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame = frame[:, :, :3]  # Convert RGBA to RGB
        writer.append_data(frame)

        for ax in ax_imgs[:num_cams]:
            ax.cla()
            ax.axis("off")

plt.close(fig)
print(f"Saved video to {output_video}")
