import data_labeling_pipeline.rhe as rhe_module
import numpy as np
import pytest
from data_labeling_pipeline.rhe import RecedingHorizonEstimator


class DummyResult:
    def __init__(self, x, success=True, message="ok"):
        self.x = np.array(x, dtype=float)
        self.success = success
        self.message = message


def test_dynamics_respects_enu_heading_convention():
    estimator = RecedingHorizonEstimator(dt=0.2)

    east_facing = estimator.dynamics(
        state=np.array([1.0, 2.0, 0.0]),
        u=np.array([3.0, 0.5]),
        m=np.array([1.0, 1.0, 0.0]),
    )
    north_facing = estimator.dynamics(
        state=np.array([1.0, 2.0, np.pi / 2]),
        u=np.array([3.0, 0.5]),
        m=np.array([1.0, 1.0, 0.0]),
    )

    assert np.allclose(east_facing, np.array([1.6, 2.0, 0.1]))
    assert np.allclose(north_facing, np.array([1.0, 2.6, np.pi / 2 + 0.1]))


def test_simulate_keeps_orientation_constant_when_omega_is_zero():
    estimator = RecedingHorizonEstimator(dt=0.1, N=4)
    x0 = np.array([0.0, 0.0, -np.pi / 3])
    U = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    m = np.array([1.0, 1.0, 0.0])

    X = estimator.simulate(x0, U, m)

    assert X.shape == (4, 3)
    assert np.allclose(X[:, 2], -np.pi / 3)
    assert np.all(np.diff(X[:, 0]) > 0)


def test_cost_wraps_heading_error_across_minus_pi_pi_boundary():
    estimator = RecedingHorizonEstimator(
        dt=0.1, N=3, P_x=np.eye(3), P_m=np.eye(3), P_w=np.zeros((3, 3))
    )
    x_guess = np.array([0.0, 0.0, np.pi - 0.02])
    m_guess = np.array([1.0, 1.0, 0.0])
    opt_vars = np.array([0.0, 0.0, -np.pi + 0.02, 1.0, 1.0, 0.0])
    U = np.zeros((3, 2))
    Z = np.zeros((3, 3))

    cost = estimator.cost(opt_vars, U, Z, x_guess, m_guess)

    assert cost == pytest.approx((0.04) ** 2, rel=1e-6)


def test_cost_treats_sensor_orientation_offset_as_separate_parameter():
    estimator = RecedingHorizonEstimator(
        dt=0.1, N=4, P_x=np.zeros((3, 3)), P_m=np.zeros((3, 3)), P_w=np.eye(3)
    )
    x0 = np.array([0.0, 0.0, 0.0])
    U = np.zeros((4, 2))
    m = np.array([1.0, 1.0, 0.5])

    X = estimator.simulate(x0, U, m)
    Z = X.copy()
    Z[:, 2] += 0.5

    opt_vars = np.concatenate((x0, m))
    cost = estimator.cost(opt_vars, U, Z, x0, m)

    assert cost == pytest.approx(0.0, abs=1e-12)


def test_estimate_uses_warm_start_and_updates_previous_orientation(monkeypatch):
    estimator = RecedingHorizonEstimator(dt=0.1, N=4)
    U = np.ones((4, 2))
    Z = np.array([
        [1.0, 2.0, 0.25],
        [1.0, 2.0, 0.25],
        [1.0, 2.0, 0.25],
        [1.0, 2.0, 0.25],
    ])

    calls = {}

    def fake_minimize(fun, x0, args, method, bounds):
        calls["x0"] = np.array(x0, dtype=float)
        calls["args"] = args
        calls["method"] = method
        calls["bounds"] = bounds
        return DummyResult([1.0, 2.0, 0.25, 0.8, 0.7, -0.2])

    monkeypatch.setattr(rhe_module, "minimize", fake_minimize)

    X_opt, m_opt = estimator.estimate(U, Z)

    assert np.allclose(calls["x0"], np.array([1.0, 2.0, 0.25, 1.0, 1.0, 0.0]))
    assert calls["method"] == "SLSQP"
    assert calls["bounds"][5] == (-np.pi, np.pi)
    assert np.allclose(m_opt, np.array([0.8, 0.7, -0.2]))
    assert estimator.prev_m is not None
    assert estimator.prev_x is not None
    assert np.allclose(estimator.prev_m, m_opt)
    assert np.allclose(estimator.prev_x, X_opt[1])
    assert np.isclose(estimator.prev_x[2], 0.32)


def test_estimate_reuses_previous_orientation_solution_on_next_call(monkeypatch):
    estimator = RecedingHorizonEstimator(dt=0.1, N=3)
    U = np.ones((3, 2))
    Z = np.zeros((3, 3))

    warm_starts = []

    def fake_minimize(fun, x0, args, method, bounds):
        warm_starts.append(np.array(x0, dtype=float))
        if len(warm_starts) == 1:
            return DummyResult([0.0, 0.0, 0.0, 0.9, 0.8, 0.1])
        return DummyResult([0.0, 0.0, 0.0, 0.7, 0.6, 0.2])

    monkeypatch.setattr(rhe_module, "minimize", fake_minimize)

    estimator.estimate(U, Z)
    estimator.estimate(U, Z)

    assert np.allclose(warm_starts[0], np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]))
    assert np.allclose(warm_starts[1], np.array([0.09, 0.0, 0.08, 0.9, 0.8, 0.1]))


def test_estimate_raises_when_optimizer_fails(monkeypatch):
    estimator = RecedingHorizonEstimator(dt=0.1, N=4)
    U = np.zeros((4, 2))
    Z = np.zeros((4, 3))

    def fake_minimize(*args, **kwargs):
        return DummyResult([0.0] * 6, success=False, message="no convergence")

    monkeypatch.setattr(rhe_module, "minimize", fake_minimize)

    with pytest.raises(ValueError, match="Optimization failed"):
        estimator.estimate(U, Z)


def test_reset_clears_state():
    estimator = RecedingHorizonEstimator()
    estimator.prev_x = np.array([1.0, 2.0, 3.0])
    estimator.prev_m = np.array([0.1, 0.2, 0.3])

    estimator.reset()

    assert estimator.prev_x is None
    assert estimator.prev_m is None


def test_simulate_with_different_parameters():
    """Test that simulate produces correct trajectory shapes with various parameters."""
    estimator = RecedingHorizonEstimator(dt=0.1, N=50)
    x0 = np.array([0.0, 0.0, 0.0])
    U = np.ones((50, 2))  # constant velocity and turning

    # Test with different parameter combinations
    param_sets = [
        np.array([1.0, 1.0, 0.0]),  # ideal
        np.array([0.5, 1.0, 0.0]),  # degraded longitudinal
        np.array([1.0, 0.5, 0.0]),  # degraded angular
        np.array([0.7, 0.8, 0.1]),  # all degraded with bias
    ]

    for params in param_sets:
        X = estimator.simulate(x0, U, params)
        assert X.shape == (50, 3)
        assert np.allclose(X[0], x0)


def test_simulate_trajectory_distance_scales_with_velocity():
    """Test that trajectory distance is proportional to velocity magnitude."""
    estimator = RecedingHorizonEstimator(dt=0.1, N=100)
    x0 = np.array([0.0, 0.0, 0.0])
    m = np.array([1.0, 1.0, 0.0])

    # Create inputs with different velocities
    U_low = np.zeros((100, 2))
    U_low[:, 0] = 1.0  # v = 1.0 m/s

    U_high = np.zeros((100, 2))
    U_high[:, 0] = 2.0  # v = 2.0 m/s

    X_low = estimator.simulate(x0, U_low, m)
    X_high = estimator.simulate(x0, U_high, m)

    dist_low = np.linalg.norm(X_low[-1, :2] - x0[:2])
    dist_high = np.linalg.norm(X_high[-1, :2] - x0[:2])

    # Distance should approximately double when velocity doubles
    assert np.isclose(dist_high / dist_low, 2.0, rtol=0.01)


def test_simulate_degraded_parameters_reduce_distance():
    """Test that degraded parameters (mu < 1) result in shorter trajectories."""
    estimator = RecedingHorizonEstimator(dt=0.1, N=100)
    x0 = np.array([0.0, 0.0, 0.0])
    U = np.ones((100, 2))

    m_ideal = np.array([1.0, 1.0, 0.0])
    m_degraded = np.array([0.5, 1.0, 0.0])

    X_ideal = estimator.simulate(x0, U, m_ideal)
    X_degraded = estimator.simulate(x0, U, m_degraded)

    dist_ideal = np.linalg.norm(X_ideal[-1, :2] - x0[:2])
    dist_degraded = np.linalg.norm(X_degraded[-1, :2] - x0[:2])

    # Degraded should travel roughly half the distance
    assert dist_degraded < dist_ideal
    assert np.isclose(dist_degraded / dist_ideal, 0.5, rtol=0.01)


def test_cost_function_with_perfect_measurement():
    """Test that cost is zero when measurement matches simulation perfectly."""
    estimator = RecedingHorizonEstimator(
        dt=0.1, N=10, P_x=np.eye(3), P_m=np.eye(3), P_w=np.eye(3)
    )
    x0 = np.array([0.0, 0.0, 0.0])
    m = np.array([1.0, 1.0, 0.0])
    U = np.ones((10, 2))

    # Simulate perfect trajectory
    X = estimator.simulate(x0, U, m)
    Z = X.copy()

    # Cost should be zero with perfect match
    opt_vars = np.concatenate((x0, m))
    cost = estimator.cost(opt_vars, U, Z, x0, m)

    assert cost == pytest.approx(0.0, abs=1e-12)


def test_covariance_matrix_defaults():
    """Test that covariance matrices are properly initialized."""
    estimator = RecedingHorizonEstimator(dt=0.1, N=10)

    assert estimator.P_x.shape == (3, 3)
    assert estimator.P_m.shape == (3, 3)
    assert estimator.P_w.shape == (3, 3)
    assert np.allclose(estimator.P_x, np.diag([1.0, 1.0, 1.0]))
    assert np.allclose(estimator.P_m, np.diag([1.0, 1.0, 1.0]))
    assert np.allclose(estimator.P_w, np.diag([1.0, 1.0, 1.0]))
