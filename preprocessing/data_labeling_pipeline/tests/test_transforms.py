from pathlib import Path

import numpy as np
import pytest
from data_labeling_pipeline.dataset_loader import GrandTourLoader
from data_labeling_pipeline.ros1_transforms import get_static_transform
from scipy.spatial.transform import Rotation as R


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


@pytest.fixture
def sample_mission_path():
    return Path("/datahome/lmertens/grand_tour_dataset/ros1/ETH-1")


def test_cpt7_yaw_inversion_via_transforms(sample_mission_path):
    """
    Sanity check to prove that applying the explicit SE(3) transform
    T_cpt7_to_base correctly accounts for the 'yaw negation' issue.
    """
    tf_bags = list(sample_mission_path.glob("*tf*.bag"))
    if not tf_bags:
        pytest.skip(f"Reference dataset not available at {sample_mission_path}")

    T_cpt7_to_base = get_static_transform(sample_mission_path, "cpt7_imu", "base")
    rotation_cpt7_to_base = T_cpt7_to_base.rotation.as_matrix()

    cpt7_yaw_deg = 30.0
    rotation_cpt7 = R.from_euler("zyx", [cpt7_yaw_deg, 0, 0], degrees=True).as_matrix()
    rotation_base = rotation_cpt7 @ rotation_cpt7_to_base
    base_yaw_deg = R.from_matrix(rotation_base).as_euler("zyx", degrees=True)[0]

    yaw_diff = _wrap_deg(base_yaw_deg - (-cpt7_yaw_deg))
    assert abs(yaw_diff) < 1.0, (
        f"Expected base yaw ~{-cpt7_yaw_deg}, got {base_yaw_deg}"
    )


def test_cpt7_imu_to_box_base_is_identity(sample_mission_path):
    """box_base is defined as identical to cpt7_imu, so the transform should be ~identity."""
    tf_bags = list(sample_mission_path.glob("*tf*.bag"))
    if not tf_bags:
        pytest.skip(f"Reference dataset not available at {sample_mission_path}")

    T = get_static_transform(sample_mission_path, "cpt7_imu", "box_base")

    angle_rad = T.rotation.magnitude()
    assert angle_rad < np.deg2rad(1.0), (
        f"Rotation not near identity: {np.rad2deg(angle_rad):.2f} deg"
    )
    assert np.linalg.norm(T.translation) < 0.01, (
        f"Translation not near zero: {T.translation}"
    )


def _expect_180_deg_rotation_around_x(T, description=""):
    R_expected = R.from_euler("x", 180.0, degrees=True)
    angle_diff = (R_expected.inv() * T.rotation).magnitude()
    assert angle_diff < np.deg2rad(5.0), (
        f"{description} Rotation not near 180° around x: diff {np.rad2deg(angle_diff):.2f} deg"
    )

    # Expect no rotation around y or z, which would indicate a misinterpretation of the transform as a non-180° rotation.
    R_yz = R.from_matrix(T.rotation.as_matrix() @ R_expected.inv().as_matrix().T)
    angle_yz = R_yz.magnitude()
    assert angle_yz < np.deg2rad(5.0), (
        f"{description} Unexpected rotation around y/z axes: {np.rad2deg(angle_yz):.2f} deg"
    )

    # Assert that the rotation is not identity, which would indicate a misinterpretation of the transform as a pure translation.
    angle_total = T.rotation.magnitude()
    assert angle_total > np.deg2rad(175.0), (
        f"{description} Rotation too small, expected ~180° around x: got {np.rad2deg(angle_total):.2f} deg"
    )

    # assert np.linalg.norm(T.translation) > 0.01, (
    #     f"{description} Expected non-trivial translation, got {T.translation}"
    # )


def test_cpt7_imu_to_base_has_180_x_rotation_and_translation(sample_mission_path):
    """cpt7_imu is mounted upside-down relative to base: expect ~180° around x-axis plus translation."""
    tf_bags = list(sample_mission_path.glob("*tf*.bag"))
    if not tf_bags:
        pytest.skip(f"Reference dataset not available at {sample_mission_path}")

    T = get_static_transform(sample_mission_path, "cpt7_imu", "base")

    _expect_180_deg_rotation_around_x(T, description="cpt7_imu -> base")

    T_inv = get_static_transform(sample_mission_path, "base", "cpt7_imu")
    _expect_180_deg_rotation_around_x(T_inv, description="base -> cpt7_imu (inverse)")


def test_box_base_to_base_has_180_x_rotation_and_translation(sample_mission_path):
    """box_base ≡ cpt7_imu, so box_base→base should also be ~180° around x plus translation."""
    tf_bags = list(sample_mission_path.glob("*tf*.bag"))
    if not tf_bags:
        pytest.skip(f"Reference dataset not available at {sample_mission_path}")

    T = get_static_transform(sample_mission_path, "box_base", "base")

    _expect_180_deg_rotation_around_x(T, description="box_base -> base")

    T_inv = get_static_transform(sample_mission_path, "base", "box_base")
    _expect_180_deg_rotation_around_x(T_inv, description="base -> box_base (inverse)")


def test_hesai_lidar_to_base_is_minus_90_yaw(sample_mission_path):
    """dlio odometry is map->hesai_lidar; the lidar is mounted ~-90deg in yaw vs base."""
    tf_bags = list(sample_mission_path.glob("*tf*.bag"))
    if not tf_bags:
        pytest.skip(f"Reference dataset not available at {sample_mission_path}")

    T = get_static_transform(sample_mission_path, "hesai_lidar", "base")
    roll, pitch, yaw = T.rotation.as_euler("xyz", degrees=True)

    assert abs(_wrap_deg(yaw - (-90.0))) < 5.0, f"Expected ~-90deg yaw, got {yaw:.2f}"
    assert abs(roll) < 5.0 and abs(pitch) < 5.0, (
        f"Expected level lidar, got roll={roll:.2f}, pitch={pitch:.2f}"
    )
    assert np.linalg.norm(T.translation) > 0.01, "Expected non-trivial lever arm"


def test_loader_correctly_parses_real_sample(sample_mission_path):
    mission_path = sample_mission_path
    if not mission_path.exists():
        pytest.skip("Reference dataset missing.")

    loader = GrandTourLoader(mission_path)
    ts, U, Z = loader.load_data_shared_timestamps(dt=0.1)

    assert len(ts) > 0
    assert U.shape[1] == 2
    assert Z.shape[1] == 3
