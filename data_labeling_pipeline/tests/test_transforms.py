import pytest
import zarr
from data_labeling_pipeline.zarr_transforms import get_static_transform
from scipy.spatial.transform import Rotation as R


@pytest.fixture
def sample_mission_path():
    # Use the reference mission for testing real transforms
    return "/datahome/lmertens/grand_tour_dataset/2024-10-01-11-29-55/data"


def test_cpt7_yaw_inversion_via_transforms(sample_mission_path):
    """
    Sanity check to prove that applying the explicit SE(3) transform
    T_cpt7_to_base correctly accounts for the 'yaw negation' issue.
    """
    try:
        tf_group = zarr.open(f"{sample_mission_path}/tf", mode="r")
    except Exception:
        pytest.skip(f"Reference dataset not available at {sample_mission_path}")

    # 1. Get the static transform from cpt7_imu to base
    T_cpt7_to_base = get_static_transform({"tf": tf_group}, "cpt7_imu", "base")

    print("T_cpt7_to_base:\n", T_cpt7_to_base)

    # 2. Verify that this transform has a ~180 degree rotation around Z,
    # which effectively inverts X and Y.
    # Note: R.from_matrix can suffer from gimbal lock at 180 degrees, but
    # the underlying math will show X and Y are inverted.
    R_cb = T_cpt7_to_base[:3, :3]

    # 3. Simulate a heading from cpt7.
    # Suppose cpt7_imu points 30 degrees counter-clockwise
    test_cpt7_yaw_deg = 30.0
    r_cpt7 = R.from_euler("zyx", [test_cpt7_yaw_deg, 0, 0], degrees=True).as_matrix()

    # Apply transform: R_base = R_cpt7 @ R_cpt7_to_base
    r_base = r_cpt7 @ R_cb

    # Extract yaw of the base
    base_yaw_deg = R.from_matrix(r_base).as_euler("zyx", degrees=True)[0]

    # The expected base yaw is -30 degrees (modulo 360) or ~330 because
    # cpt7_imu is mounted inverted (roll=-180) meaning its yaw is effectively negated!
    # Let's handle angle wrapping gracefully:
    def wrap180(ang):
        return (ang + 180) % 360 - 180

    yaw_diff = wrap180(base_yaw_deg - (-test_cpt7_yaw_deg))
    assert abs(yaw_diff) < 1.0, (
        f"Expected base yaw ~{-test_cpt7_yaw_deg}, got {base_yaw_deg}"
    )


def test_loader_correctly_parses_real_sample(sample_mission_path):
    """
    Integration test using the GrandTourLoader to ensure the
    transformation logic works end-to-end without crashing.
    """
    import pathlib

    from data_labeling_pipeline.dataset_loader import GrandTourLoader

    mission_path = pathlib.Path(sample_mission_path).parent
    if not mission_path.exists():
        pytest.skip("Reference dataset missing.")

    loader = GrandTourLoader(mission_path)
    # Just loading one sample is enough to verify logic runs without error
    ts, U, Z = loader.load_data_shared_timestamps(dt=0.1)

    assert len(ts) > 0
    assert U.shape[1] == 2
    assert Z.shape[1] == 3


if __name__ == "__main__":
    test_cpt7_yaw_inversion_via_transforms(
        "/datahome/lmertens/grand_tour_dataset/2024-10-01-11-29-55/data"
    )
