import numpy as np
import pytest
from data_labeling_pipeline.dataset_loader import (
    fill_command_gaps_with_zero_commands,
    quaternions_to_yaws,
)
from scipy.spatial.transform import Rotation


@pytest.mark.parametrize(
    ("yaw_deg", "expected_yaw"),
    [
        (0.0, 0.0),
        (90.0, np.pi / 2),
        (-90.0, -np.pi / 2),
        (45.0, np.pi / 4),
    ],
)
def test_quaternions_to_yaws(yaw_deg, expected_yaw):
    quat_wxyz = Rotation.from_euler("z", yaw_deg, degrees=True).as_quat(
        scalar_first=True
    )
    computed = quaternions_to_yaws([quat_wxyz], scalar_first=True)
    assert computed == pytest.approx([expected_yaw], abs=1e-4)


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
