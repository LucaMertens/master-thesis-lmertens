import numpy as np
from data_labeling_pipeline.dataset_loader import (
    fill_command_gaps_with_zero_commands,
    quaternions_to_yaws,
)
from scipy.spatial.transform import Rotation


def test_quaternions_to_yaws():
    # Test with known quaternions and expected yaw angles
    quaternions_wxyz = [
        # No rotation -> yaw = 0
        [1, 0, 0, 0],
        # 90 deg around Z -> yaw = 90
        Rotation.from_euler("z", 90, degrees=True).as_quat(scalar_first=True),
        # -90 deg around Z -> yaw = -90
        Rotation.from_euler("z", -90, degrees=True).as_quat(scalar_first=True),
        # 45 deg around Z -> yaw = 45
        Rotation.from_euler("z", 45, degrees=True).as_quat(scalar_first=True),
    ]

    expected_yaws = [0, np.pi / 2, -np.pi / 2, np.pi / 4]

    computed_yaws = quaternions_to_yaws(quaternions_wxyz, scalar_first=True)

    assert np.allclose(computed_yaws, expected_yaws, atol=1e-4), (
        f"Failed: expected {expected_yaws}, got {computed_yaws}"
    )
    print("Test passed: Quaternion to yaw conversion is correct.")


def test_fill_command_gaps_with_zero_commands_inserts_missing_samples():
    timestamps = np.array([0.0, 0.1, 0.2, 0.6, 0.7])
    linear = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    angular = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])

    filled_ts, filled_linear, filled_angular = fill_command_gaps_with_zero_commands(
        timestamps, linear, angular, gap_factor=3.0
    )

    expected_ts = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    expected_linear = np.array([[1.0], [2.0], [3.0], [0.0], [0.0], [0.0], [4.0], [5.0]])
    expected_angular = np.array([
        [10.0],
        [20.0],
        [30.0],
        [0.0],
        [0.0],
        [0.0],
        [40.0],
        [50.0],
    ])

    assert np.allclose(filled_ts, expected_ts)
    assert np.allclose(filled_linear, expected_linear)
    assert np.allclose(filled_angular, expected_angular)
