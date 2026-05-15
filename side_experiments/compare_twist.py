import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent / "data_labeling_pipeline" / "src"))

from data_labeling_pipeline.dataset_loader import GrandTourLoader
from data_labeling_pipeline.ros1_transforms import (
    get_static_transform,
    get_transform_series,
)


def compare_twist(mission_shortname):
    mission_path = (
        Path("/datahome/lmertens/grand_tour_dataset/ros1/") / mission_shortname
    )
    print(f"Loading data from {mission_path}...")
    loader = GrandTourLoader(mission_path)

    # Load commands (base frame)
    try:
        cmd_ts, cmd_linear, cmd_angular = loader.load_commands()
        print(f"Loaded {len(cmd_ts)} command messages.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load odometry (cpt7_imu frame)
    try:
        odom_ts, odom_pos, odom_orien, odom_linear, odom_angular = (
            loader.load_odometry()
        )
        print(f"Loaded {len(odom_ts)} odometry messages.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Get static transform from cpt7_imu to base
    try:
        T_cpt7_to_base = get_static_transform(mission_path, "cpt7_imu", "base")
        print("Found static transform cpt7_imu -> base")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Transform odometry twist to base frame
    # Assumption: The provided twist is in the world frame (base_enu/enu_origin)
    # We need to rotate it to the robot's body frame (base).
    odom_linear_base = []
    odom_angular_base = []

    # R_cpt7_to_base is constant
    R_cpt7_to_base = T_cpt7_to_base.rotation
    interpolator = get_transform_series(mission_path, "enu_origin", "cpt7_imu")

    for i in range(len(odom_linear)):
        v_world = odom_linear[i]
        w_world = odom_angular[i]

        # get R_enu_to_base from interpolator
        R_enu_to_base = interpolator(odom_ts[i]).rotation * T_cpt7_to_base.rotation

        # Rotate linear velocity from world to body frame
        v_b = R_enu_to_base.apply(v_world)
        w_b = R_enu_to_base.apply(w_world)

        odom_linear_base.append(v_b)
        odom_angular_base.append(w_b)

    odom_linear_base = np.array(odom_linear_base)
    odom_angular_base = np.array(odom_angular_base)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    # Linear velocities
    for i, label in enumerate(["X", "Y", "Z"]):
        axes[0, i].plot(
            cmd_ts - cmd_ts[0], cmd_linear[:, i], label="Command (base)", alpha=0.7
        )
        axes[0, i].plot(
            odom_ts - cmd_ts[0], odom_linear_base[:, i], label="Odom (base)", alpha=0.7
        )
        axes[0, i].set_title(f"Linear Velocity {label}")
        axes[0, i].legend()
        axes[0, i].set_ylabel("m/s")
        axes[0, i].grid(True)

    # Angular velocities
    for i, label in enumerate(["X", "Y", "Z"]):
        axes[1, i].plot(
            cmd_ts - cmd_ts[0], cmd_angular[:, i], label="Command (base)", alpha=0.7
        )
        axes[1, i].plot(
            odom_ts - cmd_ts[0], odom_angular_base[:, i], label="Odom (base)", alpha=0.7
        )
        axes[1, i].set_title(f"Angular Velocity {label}")
        axes[1, i].legend()
        axes[1, i].set_ylabel("rad/s")
        axes[1, i].set_xlabel("Time [s]")
        axes[1, i].grid(True)

    plt.tight_layout()
    plot_path = Path("twist_comparison.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

    # Demonstrate usage of non-static transform as requested
    # try:
    print("\nChecking non-static transforms...")
    # The user mentioned cpt7-imu --> enu_origin in .tf
    interpolator = get_transform_series(mission_path, "enu_origin", "cpt7_imu")
    print(
        f"Found {len(interpolator.times)} non-static transform entries for enu_origin -> cpt7_imu"
    )

    # Verify if odometry pose matches TF
    t_test = odom_ts[len(odom_ts) // 2]
    T_tf = interpolator(t_test)

    # Find closest odom pose
    idx = np.argmin(np.abs(odom_ts - t_test))
    p_odom = odom_pos[idx]

    print(f"At t={t_test - odom_ts[0]:.2f}s:")
    print(f"  TF translation:   {T_tf.translation}")
    print(f"  Odom translation: {p_odom}")
    diff = np.linalg.norm(T_tf.translation - p_odom)
    print(f"  Position difference: {diff:.4f}m")

    # except Exception as e:
    #     print(f"Non-static transform check skipped or failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_twist.py <mission_path>")
        sys.exit(1)
    compare_twist(sys.argv[1])
