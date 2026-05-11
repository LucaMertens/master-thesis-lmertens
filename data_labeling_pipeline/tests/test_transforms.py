from pathlib import Path

import pytest
import zarr
from data_labeling_pipeline.dataset_loader import GrandTourLoader
from data_labeling_pipeline.zarr_transforms import get_static_transform
from scipy.spatial.transform import Rotation as R


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


@pytest.fixture
def sample_mission_path():
    return Path("/datahome/lmertens/grand_tour_dataset/2024-10-01-11-29-55/data")


def test_cpt7_yaw_inversion_via_transforms(sample_mission_path):
    """
    Sanity check to prove that applying the explicit SE(3) transform
    T_cpt7_to_base correctly accounts for the 'yaw negation' issue.
    """
    try:
        tf_group = zarr.open(str(sample_mission_path / "tf"), mode="r")
    except Exception:
        pytest.skip(f"Reference dataset not available at {sample_mission_path}")

    T_cpt7_to_base = get_static_transform({"tf": tf_group}, "cpt7_imu", "base")
    rotation_cpt7_to_base = T_cpt7_to_base[:3, :3]

    cpt7_yaw_deg = 30.0
    rotation_cpt7 = R.from_euler("zyx", [cpt7_yaw_deg, 0, 0], degrees=True).as_matrix()
    rotation_base = rotation_cpt7 @ rotation_cpt7_to_base
    base_yaw_deg = R.from_matrix(rotation_base).as_euler("zyx", degrees=True)[0]

    yaw_diff = _wrap_deg(base_yaw_deg - (-cpt7_yaw_deg))
    assert abs(yaw_diff) < 1.0, (
        f"Expected base yaw ~{-cpt7_yaw_deg}, got {base_yaw_deg}"
    )


def test_loader_correctly_parses_real_sample(sample_mission_path):
    mission_path = sample_mission_path.parent
    if not mission_path.exists():
        pytest.skip("Reference dataset missing.")

    loader = GrandTourLoader(mission_path)
    ts, U, Z = loader.load_data_shared_timestamps(dt=0.1)

    assert len(ts) > 0
    assert U.shape[1] == 2
    assert Z.shape[1] == 3
