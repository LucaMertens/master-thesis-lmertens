import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent / "preprocessing" / "data_labeling_pipeline" / "src"))

from data_labeling_pipeline.dataset_loader import GrandTourLoader
from data_labeling_pipeline.ros1_transforms import get_static_transform

# The /boxi/inertial_explorer/tc/odometry twist is expressed in NED (North-East-Down),
# not in ENU or body frame, despite what the header frames suggest.
# The NovAtel Inertial Explorer TC exporter outputs velocity in its native NED navigation
# frame. Angular velocity is not exported (always zero); it is derived here from the
# pose orientation time series.
_R_NED_TO_ENU = R.from_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float))


def ned_twist_to_base(odom_linear, odom_orien, T_cpt7_to_base):
    """Convert odom linear velocity from NED -> ENU -> cpt7_imu -> base."""
    x = odom_linear
    R_enu_to_cpt7 = R.from_quat(odom_orien).inv()
    R_cpt7_to_base = T_cpt7_to_base.rotation

    x = _R_NED_TO_ENU.apply(x)
    x = R_enu_to_cpt7.apply(x)
    x = R_cpt7_to_base.apply(x)

    return x


def derive_angular_velocity_base(odom_ts, odom_orien, T_cpt7_to_base):
    """Numerically differentiate pose orientation to get angular velocity in base frame.

    Returns (ts_mid, omega_base) where ts_mid are midpoint timestamps.
    """
    rots = R.from_quat(odom_orien)
    dt = np.diff(odom_ts)
    valid = dt > 1e-9

    # Body-frame angular velocity: dR_body = R(t).inv() * R(t+dt), rotvec / dt
    R_prev = rots[:-1][valid]
    R_next = rots[1:][valid]
    dR_body = R_prev.inv() * R_next
    omega_cpt7 = dR_body.as_rotvec() / dt[valid, None]

    omega_base = T_cpt7_to_base.rotation.apply(omega_cpt7)
    ts_mid = 0.5 * (odom_ts[:-1][valid] + odom_ts[1:][valid])
    return ts_mid, omega_base


def compare_twist(mission_shortname):
    mission_path = (
        Path("/datahome/lmertens/grand_tour_dataset/ros1/") / mission_shortname
    )
    print(f"Loading data from {mission_path}...")
    loader = GrandTourLoader(mission_path)

    cmd_ts, cmd_linear, cmd_angular = loader.load_commands()
    print(f"Loaded {len(cmd_ts)} command messages.")

    odom_ts, _, odom_orien, odom_linear, _ = loader.load_odometry()
    print(f"Loaded {len(odom_ts)} odometry messages.")

    T_cpt7_to_base = get_static_transform(mission_path, "cpt7_imu", "base")
    print("Found static transform cpt7_imu -> base")

    # NED -> ENU -> cpt7_imu -> base
    odom_linear_base = ned_twist_to_base(odom_linear, odom_orien, T_cpt7_to_base)

    # Angular velocity derived from pose differentiation (TC odom has none)
    ang_ts, odom_angular_base = derive_angular_velocity_base(
        odom_ts, odom_orien, T_cpt7_to_base
    )

    # Quick stats
    t0 = max(cmd_ts[0], odom_ts[0])
    t1 = min(cmd_ts[-1], odom_ts[-1])
    mask = (odom_ts >= t0) & (odom_ts <= t1)
    cmd_x_at_odom = np.interp(odom_ts[mask], cmd_ts, cmd_linear[:, 0])
    moving = np.abs(cmd_x_at_odom) > 0.05
    vb = odom_linear_base[mask]
    if moving.any():
        mse_x = np.mean((vb[moving, 0] - cmd_x_at_odom[moving]) ** 2)
        corr_x = float(np.corrcoef(vb[moving, 0], cmd_x_at_odom[moving])[0, 1])
        print(
            f"\nForward velocity (moving samples): MSE_x={mse_x:.4f}, Corr_x={corr_x:.4f}"
        )

    # Plotting
    _, axes = plt.subplots(5, 3, figsize=(18, 10), sharex=True)
    t_ref = cmd_ts[0]

    for i, label in enumerate(["X (forward)", "Y (lateral)", "Z (vertical)"]):
        axes[0, i].plot(
            cmd_ts - t_ref, cmd_linear[:, i], label="Command (base)", alpha=0.8
        )
        axes[0, i].plot(
            odom_ts - t_ref, odom_linear_base[:, i], label="Odom (base)", alpha=0.6
        )
        axes[0, i].set_title(f"Linear Velocity {label}")
        axes[0, i].legend()
        axes[0, i].set_ylabel("m/s")
        axes[0, i].grid(True)

    for i, label in enumerate(["X (roll rate)", "Y (pitch rate)", "Z (yaw rate)"]):
        axes[1, i].plot(
            cmd_ts - t_ref, cmd_angular[:, i], label="Command (base)", alpha=0.8
        )
        axes[1, i].plot(
            ang_ts - t_ref, odom_angular_base[:, i], label="Odom (derived)", alpha=0.6
        )
        axes[1, i].set_title(f"Angular Velocity {label}")
        axes[1, i].legend()
        axes[1, i].set_ylabel("rad/s")
        axes[1, i].set_xlabel("Time [s]")
        axes[1, i].grid(True)

    curr_row = 2

    # Raw NED odom
    for i, label in enumerate(["X (forward)", "Y (lateral)", "Z (vertical)"]):
        axes[curr_row, i].plot(
            odom_ts - t_ref,
            odom_linear[:, i],
            label="Odom (NED)",
            alpha=0.6,
            color="orange",
        )
        axes[curr_row, i].set_title(f"Raw Odom Linear Velocity {label} (NED)")
        axes[curr_row, i].legend()
        axes[curr_row, i].set_ylabel("m/s")
        axes[curr_row, i].set_xlabel("Time [s]")
        axes[curr_row, i].grid(True)

    curr_row += 1
    x = _R_NED_TO_ENU.apply(odom_linear)
    for i, label in enumerate(["X (forward)", "Y (lateral)", "Z (vertical)"]):
        axes[curr_row, i].plot(
            odom_ts - t_ref,
            x[:, i],
            label="Odom (ENU)",
            alpha=0.6,
            color="green",
        )
        axes[curr_row, i].set_title(f"Odom Linear Velocity {label} in ENU Frame")
        axes[curr_row, i].legend()
        axes[curr_row, i].set_ylabel("m/s")
        axes[curr_row, i].set_xlabel("Time [s]")
        axes[curr_row, i].grid(True)

    plt.suptitle(f"Mission {mission_shortname}: Command vs Odometry Twist (base frame)")
    plt.tight_layout()
    plot_path = Path(f"twist_comparison_{mission_shortname}.png")
    plt.savefig(plot_path, dpi=100)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mission_shortname = sys.argv[1]
    else:
        mission_shortname = "ALB-1"
    compare_twist(mission_shortname)
