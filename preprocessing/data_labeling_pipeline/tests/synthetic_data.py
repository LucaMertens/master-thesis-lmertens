"""Synthetic scenarios for RHE benchmarks."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from data_labeling_pipeline.rhe import RecedingHorizonEstimator


@dataclass
class SyntheticScenario:
    """Container for a single synthetic benchmark scenario."""

    name: str
    description: str
    dt: float
    N_total: int
    U: np.ndarray  # (N_total, 2) commanded inputs [v, omega]
    Z: np.ndarray  # (N_total, 3) noisy measurements [px, py, theta]
    X_true: np.ndarray  # (N_total, 3) true trajectory [px, py, theta]
    m_true: np.ndarray  # (N_total, 3) per-step true params [mu, nu, delta_theta]
    noise_sigma: float = 0.0  # measurement noise std


def _generate_trajectory(
    U: np.ndarray,
    m_segments: List[Tuple[int, np.ndarray]],
    dt: float,
    x0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a trajectory and per-step parameters with piecewise terrain."""
    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.0])

    N_total = U.shape[0]
    rhe = RecedingHorizonEstimator(dt=dt, N=N_total)

    X_true = np.zeros((N_total, 3))
    m_true = np.zeros((N_total, 3))
    X_true[0] = x0

    seg_idx = 0
    for i in range(N_total):
        while seg_idx < len(m_segments) - 1 and i >= m_segments[seg_idx][0]:
            seg_idx += 1
        m_true[i] = m_segments[seg_idx][1]

    for i in range(N_total - 1):
        X_true[i + 1] = rhe.dynamics(X_true[i], U[i], m_true[i])

    return X_true, m_true


def _add_noise_and_bias(
    X_true: np.ndarray,
    m_true: np.ndarray,
    noise_sigma: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian noise and heading bias to the simulated state."""
    if rng is None:
        rng = np.random.default_rng(42)

    Z = X_true.copy()

    if noise_sigma > 0:
        Z += rng.normal(0, noise_sigma, X_true.shape)

    Z[:, 2] += m_true[:, 2]

    return Z


def _sample_truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    size: Optional[int] = None,
) -> np.ndarray:
    values = rng.normal(mean, std, size=size)
    return np.clip(values, low, high)


def _generate_segmented_commands(
    rng: np.random.Generator,
    N_total: int,
) -> np.ndarray:
    """Generate piecewise-constant command segments informed by mission stats."""
    U = np.zeros((N_total, 2))
    t = 0

    while t < N_total:
        mode = rng.choice(
            [
                "stop",
                "cruise",
                "turn",
            ],
            p=[0.15, 0.55, 0.30],
        )

        if mode == "stop":
            v_cmd = 0.0
            w_cmd = 0.0
            seg_len = int(rng.integers(15, 50))
        elif mode == "cruise":
            v_cmd = float(_sample_truncated_normal(rng, 0.7, 0.25, -0.4, 1.3))
            w_cmd = 0.0
            seg_len = int(rng.integers(20, 60))
        else:
            v_cmd = float(_sample_truncated_normal(rng, 0.65, 0.22, -0.4, 1.2))
            w_cmd = float(_sample_truncated_normal(rng, -0.12, 0.25, -0.8, 0.8))
            if rng.random() < 0.2:
                v_cmd = 0.0
            seg_len = int(rng.integers(15, 45))

        end = min(N_total, t + seg_len)
        U[t:end, 0] = v_cmd
        U[t:end, 1] = w_cmd
        t = end

    return U


# =============================================================================
# Scenario generators
# =============================================================================


def scenario_constant_serpentine(
    dt: float = 0.1,
    N_total: int = 300,
    mu: float = 0.8,
    nu: float = 0.85,
    delta_theta: float = 0.05,
    noise_sigma: float = 0.02,
) -> SyntheticScenario:
    """Constant terrain with alternating turn direction."""
    rng = np.random.default_rng(42)

    U = np.zeros((N_total, 2))
    U[:, 0] = 1.0

    # Serpentine: alternate turning direction every quarter
    quarter = N_total // 4
    U[0:quarter, 1] = 0.3
    U[quarter : 2 * quarter, 1] = -0.3
    U[2 * quarter : 3 * quarter, 1] = 0.3
    U[3 * quarter :, 1] = -0.3

    m = np.array([mu, nu, delta_theta])
    m_segments = [(N_total, m)]

    X_true, m_true = _generate_trajectory(U, m_segments, dt)
    Z = _add_noise_and_bias(X_true, m_true, noise_sigma, rng)

    return SyntheticScenario(
        name="constant_serpentine",
        description=f"Constant terrain (μ={mu}, ν={nu}, Δθ={delta_theta}) with serpentine walking",
        dt=dt,
        N_total=N_total,
        U=U,
        Z=Z,
        X_true=X_true,
        m_true=m_true,
        noise_sigma=noise_sigma,
    )


def scenario_terrain_transition(
    dt: float = 0.1,
    N_total: int = 400,
    mu1: float = 0.95,
    nu1: float = 0.9,
    mu2: float = 0.5,
    nu2: float = 0.55,
    delta_theta: float = 0.03,
    noise_sigma: float = 0.02,
) -> SyntheticScenario:
    """Abrupt terrain change at midpoint."""
    rng = np.random.default_rng(42)
    mid = N_total // 2

    U = np.zeros((N_total, 2))
    U[:, 0] = 1.0
    quarter = N_total // 4
    U[0:quarter, 1] = 0.3
    U[quarter : 2 * quarter, 1] = -0.3
    U[2 * quarter : 3 * quarter, 1] = 0.3
    U[3 * quarter :, 1] = -0.3

    m1 = np.array([mu1, nu1, delta_theta])
    m2 = np.array([mu2, nu2, delta_theta])
    m_segments = [(mid, m1), (N_total, m2)]

    X_true, m_true = _generate_trajectory(U, m_segments, dt)
    Z = _add_noise_and_bias(X_true, m_true, noise_sigma, rng)

    return SyntheticScenario(
        name="terrain_transition",
        description=f"Terrain transition: (μ={mu1},ν={nu1}) → (μ={mu2},ν={nu2}) at step {mid}",
        dt=dt,
        N_total=N_total,
        U=U,
        Z=Z,
        X_true=X_true,
        m_true=m_true,
        noise_sigma=noise_sigma,
    )


def scenario_straight_walking(
    dt: float = 0.1,
    N_total: int = 300,
    mu: float = 0.8,
    nu: float = 0.85,
    delta_theta: float = 0.02,
    noise_sigma: float = 0.02,
) -> SyntheticScenario:
    """Straight walking with no angular command."""
    rng = np.random.default_rng(42)

    U = np.zeros((N_total, 2))
    U[:, 0] = 1.0

    m = np.array([mu, nu, delta_theta])
    m_segments = [(N_total, m)]

    X_true, m_true = _generate_trajectory(U, m_segments, dt)
    Z = _add_noise_and_bias(X_true, m_true, noise_sigma, rng)

    return SyntheticScenario(
        name="straight_walking",
        description=f"Straight walking (ω=0): μ={mu}, ν={nu} (ν unobservable)",
        dt=dt,
        N_total=N_total,
        U=U,
        Z=Z,
        X_true=X_true,
        m_true=m_true,
        noise_sigma=noise_sigma,
    )


def scenario_turning_in_place(
    dt: float = 0.1,
    N_total: int = 300,
    mu: float = 0.8,
    nu: float = 0.85,
    delta_theta: float = 0.02,
    noise_sigma: float = 0.02,
) -> SyntheticScenario:
    """Turning in place with no linear command."""
    rng = np.random.default_rng(42)

    U = np.zeros((N_total, 2))
    # Alternate turning direction
    half = N_total // 2
    U[:half, 1] = 0.5
    U[half:, 1] = -0.5

    m = np.array([mu, nu, delta_theta])
    m_segments = [(N_total, m)]

    X_true, m_true = _generate_trajectory(U, m_segments, dt)
    Z = _add_noise_and_bias(X_true, m_true, noise_sigma, rng)

    return SyntheticScenario(
        name="turning_in_place",
        description=f"Turning in place (v=0): μ={mu}, ν={nu} (μ unobservable)",
        dt=dt,
        N_total=N_total,
        U=U,
        Z=Z,
        X_true=X_true,
        m_true=m_true,
        noise_sigma=noise_sigma,
    )


def scenario_full_stop(
    dt: float = 0.1,
    N_total: int = 200,
    mu: float = 0.7,
    nu: float = 0.75,
    delta_theta: float = 0.01,
    noise_sigma: float = 0.02,
) -> SyntheticScenario:
    """No-motion scenario."""
    rng = np.random.default_rng(42)

    U = np.zeros((N_total, 2))

    m = np.array([mu, nu, delta_theta])
    m_segments = [(N_total, m)]

    X_true, m_true = _generate_trajectory(U, m_segments, dt)
    Z = _add_noise_and_bias(X_true, m_true, noise_sigma, rng)

    return SyntheticScenario(
        name="full_stop",
        description="Full stop (v=0, ω=0): both params unobservable",
        dt=dt,
        N_total=N_total,
        U=U,
        Z=Z,
        X_true=X_true,
        m_true=m_true,
        noise_sigma=noise_sigma,
    )


def scenario_mixed_excitation(
    dt: float = 0.1,
    N_total: int = 600,
    mu: float = 0.75,
    nu: float = 0.80,
    delta_theta: float = 0.04,
    noise_sigma: float = 0.02,
) -> SyntheticScenario:
    """
    Alternates: serpentine → straight → turning in place → serpentine.
    Tests transitions between excitation modes.
    """
    rng = np.random.default_rng(42)
    phase_len = N_total // 4

    U = np.zeros((N_total, 2))

    # Phase 1: Serpentine (full excitation)
    U[0:phase_len, 0] = 1.0
    half_phase = phase_len // 2
    U[0:half_phase, 1] = 0.3
    U[half_phase:phase_len, 1] = -0.3

    # Phase 2: Straight walking (no angular excitation)
    U[phase_len : 2 * phase_len, 0] = 1.0
    U[phase_len : 2 * phase_len, 1] = 0.0

    # Phase 3: Turning in place (no linear excitation)
    U[2 * phase_len : 3 * phase_len, 0] = 0.0
    half_phase3 = phase_len // 2
    U[2 * phase_len : 2 * phase_len + half_phase3, 1] = 0.5
    U[2 * phase_len + half_phase3 : 3 * phase_len, 1] = -0.5

    # Phase 4: Serpentine again (full excitation)
    U[3 * phase_len :, 0] = 1.0
    remaining = N_total - 3 * phase_len
    half_rem = remaining // 2
    U[3 * phase_len : 3 * phase_len + half_rem, 1] = 0.3
    U[3 * phase_len + half_rem :, 1] = -0.3

    m = np.array([mu, nu, delta_theta])
    m_segments = [(N_total, m)]

    X_true, m_true = _generate_trajectory(U, m_segments, dt)
    Z = _add_noise_and_bias(X_true, m_true, noise_sigma, rng)

    return SyntheticScenario(
        name="mixed_excitation",
        description=f"Mixed excitation: serpentine→straight→turn-in-place→serpentine (μ={mu}, ν={nu})",
        dt=dt,
        N_total=N_total,
        U=U,
        Z=Z,
        X_true=X_true,
        m_true=m_true,
        noise_sigma=noise_sigma,
    )


def scenario_realistic_mission_profile(
    dt: float = 0.1,
    N_total: int = 900,
    mu: float = 0.85,
    nu: float = 0.88,
    delta_theta: float = 0.03,
    noise_sigma: float = 0.02,
    seed: int = 42,
) -> SyntheticScenario:
    """
    Tries to mimic real mission command stats with piecewise terrain.
    Stats used as guidance: v mean ~0.56, v zero ~0.15, ω zero ~0.6, ω biased negative.
    """
    rng = np.random.default_rng(seed)

    U = _generate_segmented_commands(rng, N_total)

    seg1_end = int(0.4 * N_total)
    seg2_end = int(0.7 * N_total)
    m1 = np.array([mu, nu, delta_theta])
    m2 = np.array([0.62, 0.68, delta_theta])
    m3 = np.array([0.78, 0.82, delta_theta])
    m_segments = [(seg1_end, m1), (seg2_end, m2), (N_total, m3)]

    X_true, m_true = _generate_trajectory(U, m_segments, dt)
    Z = _add_noise_and_bias(X_true, m_true, noise_sigma, rng)

    return SyntheticScenario(
        name="realistic_mission_profile",
        description="Data-informed commands with terrain drift (real mission stats)",
        dt=dt,
        N_total=N_total,
        U=U,
        Z=Z,
        X_true=X_true,
        m_true=m_true,
        noise_sigma=noise_sigma,
    )


def scenario_step_ramp(
    dt: float = 0.1,
    N_total: int = 480,
    mu: float = 0.82,
    nu: float = 0.84,
    delta_theta: float = 0.03,
    noise_sigma: float = 0.02,
) -> SyntheticScenario:
    """Stepwise ramp in linear and angular commands."""
    rng = np.random.default_rng(42)

    U = np.zeros((N_total, 2))
    step_count = 8
    step_len = N_total // step_count

    v_steps = np.arange(step_count, dtype=float) * 0.05
    w_steps = np.arange(step_count, dtype=float) * 0.05

    for idx in range(step_count):
        start = idx * step_len
        end = N_total if idx == step_count - 1 else (idx + 1) * step_len
        U[start:end, 0] = v_steps[idx]
        U[start:end, 1] = w_steps[idx]

    m = np.array([mu, nu, delta_theta])
    m_segments = [(N_total, m)]

    X_true, m_true = _generate_trajectory(U, m_segments, dt)
    Z = _add_noise_and_bias(X_true, m_true, noise_sigma, rng)

    return SyntheticScenario(
        name="step_ramp",
        description=f"Stepwise ramp from zero command: μ={mu}, ν={nu}",
        dt=dt,
        N_total=N_total,
        U=U,
        Z=Z,
        X_true=X_true,
        m_true=m_true,
        noise_sigma=noise_sigma,
    )


def scenario_noise_robustness(
    noise_sigma: float,
    dt: float = 0.1,
    N_total: int = 300,
    mu: float = 0.8,
    nu: float = 0.85,
    delta_theta: float = 0.05,
) -> SyntheticScenario:
    """Constant serpentine scenario with configurable noise."""
    rng = np.random.default_rng(42)

    U = np.zeros((N_total, 2))
    U[:, 0] = 1.0
    quarter = N_total // 4
    U[0:quarter, 1] = 0.3
    U[quarter : 2 * quarter, 1] = -0.3
    U[2 * quarter : 3 * quarter, 1] = 0.3
    U[3 * quarter :, 1] = -0.3

    m = np.array([mu, nu, delta_theta])
    m_segments = [(N_total, m)]

    X_true, m_true = _generate_trajectory(U, m_segments, dt)
    Z = _add_noise_and_bias(X_true, m_true, noise_sigma, rng)

    return SyntheticScenario(
        name=f"noise_sigma_{noise_sigma:.3f}",
        description=f"Noise robustness (σ={noise_sigma}): μ={mu}, ν={nu}",
        dt=dt,
        N_total=N_total,
        U=U,
        Z=Z,
        X_true=X_true,
        m_true=m_true,
        noise_sigma=noise_sigma,
    )


def get_all_scenarios() -> List[SyntheticScenario]:
    """Generate all benchmark scenarios."""
    scenarios = [
        scenario_constant_serpentine(),
        scenario_terrain_transition(),
        scenario_straight_walking(),
        scenario_turning_in_place(),
        scenario_full_stop(),
        scenario_mixed_excitation(),
        scenario_realistic_mission_profile(),
        scenario_step_ramp(),
    ]

    for sigma in [0.01, 0.05, 0.1, 0.2]:
        scenarios.append(scenario_noise_robustness(noise_sigma=sigma))

    return scenarios
