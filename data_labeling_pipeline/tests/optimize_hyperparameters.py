"""
Hyperparameter optimizer for the RHE data labeling pipeline.

Uses derivative-free optimization (differential evolution) to find
hyperparameters that minimize estimation error on a realistic synthetic
scenario: linear command non-zero most of the time, angular velocity
often zero — typical quadruped behavior.

Optimized parameters:
  - P_x diagonal (3 values): initial state trust
  - P_m diagonal (3 values): parameter smoothness trust
  - P_w diagonal (3 values): measurement trust
    - lambda_max (max_tie_strengh): maximum tie strength

Fixed (not optimized):
    - v_thresh, w_thresh: confidence thresholds

Usage:
cd data_labeling_pipeline
nohup uv run python -u -c "from tests.optimize_hyperparameters import main; main()" > output.log 2>&1 &
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from data_labeling_pipeline.hyperparameters import STEP_SIZE, N, dt, v_thresh, w_thresh
from data_labeling_pipeline.rhe import RecedingHorizonEstimator
from scipy.optimize import differential_evolution

from .synthetic_data import SyntheticScenario, scenario_realistic_mission_profile


def generate_realistic_scenario(
    dt: float = 0.1,
    N_total: int = 900,
    noise_sigma: float = 0.02,
    seed: int = 42,
) -> SyntheticScenario:
    """Realistic scenario driven by data-informed command distributions."""
    return scenario_realistic_mission_profile(
        dt=dt,
        N_total=N_total,
        noise_sigma=noise_sigma,
        seed=seed,
    )
    # return scenario_mixed_excitation(dt=dt, N_total=N_total, noise_sigma=noise_sigma)


# Parameter layout in the optimization vector (reduced DOF):
#   `P_x[0]`  -> P_x[0] and P_x[1] (tied)
#   `P_x[2]`  -> P_x[2]
#   `P_m[0..2]` -> P_m diagonal (mu, nu, delta_theta)
#   `P_w[0]`  -> P_w[0] and P_w[1] (tied)
#   `P_w[2]`  -> P_w[2]
#   `lambda_max` -> max_tie_strengh

PARAM_NAMES = [
    "P_x[0]",  # tied -> sets P_x[0] and P_x[1]
    "P_x[2]",
    "P_m[0]",
    "P_m[1]",
    "P_m[2]",
    "P_w[0]",  # tied -> sets P_w[0] and P_w[1]
    "P_w[2]",
    "lambda_max",
]

# Bounds for each parameter.
# These are quadratic weights in the cost function, so they span orders of magnitude.
PARAM_BOUNDS = [
    (1e-3, 5),  # P_x[0] - sets P_x[0] & P_x[1]
    (1e-3, 5),  # P_x[2]
    (1e-3, 5),  # P_m[0] - μ
    (1e-3, 5),  # P_m[1] - ν
    (1e-3, 5),  # P_m[2] - δθ
    (1e-3, 5),  # P_w[0] - sets P_w[0] & P_w[1]
    (1e-3, 5),  # P_w[2] - θ
    (1e-2, 5),  # lambda_max (max_tie_strengh)
]


def unpack_params(x: np.ndarray) -> Dict:
    """Convert flat optimization vector → RHE constructor kwargs."""
    # tie P_x[0]==P_x[1] and P_w[0]==P_w[1]
    x = np.asarray(x).ravel()
    return {
        "P_x": np.diag([float(x[0]), float(x[0]), float(x[1])]),
        "P_m": np.diag([float(x[2]), float(x[3]), float(x[4])]),
        "P_w": np.diag([float(x[5]), float(x[5]), float(x[6])]),
        "max_tie_strengh": float(x[7]),
    }


def pack_current_params() -> np.ndarray:
    """Pack current hyperparameters into a flat vector (for reference)."""

    from data_labeling_pipeline.hyperparameters import P_m as cur_P_m
    from data_labeling_pipeline.hyperparameters import P_w as cur_P_w
    from data_labeling_pipeline.hyperparameters import P_x as cur_P_x
    from data_labeling_pipeline.hyperparameters import (
        max_tie_strengh as cur_max_tie_strengh,
    )

    tied_P_x0 = float((cur_P_x[0, 0] + cur_P_x[1, 1]) / 2.0)
    tied_P_w0 = float((cur_P_w[0, 0] + cur_P_w[1, 1]) / 2.0)

    return np.array([
        tied_P_x0,
        float(cur_P_x[2, 2]),
        float(cur_P_m[0, 0]),
        float(cur_P_m[1, 1]),
        float(cur_P_m[2, 2]),
        tied_P_w0,
        float(cur_P_w[2, 2]),
        float(cur_max_tie_strengh),
    ])


def objective(
    param_vector: np.ndarray,
    scenario: SyntheticScenario,
    horizon_n: int,
    step_size: int,
    param_weight: float = 0.7,
) -> float:
    """Objective: confidence-weighted parameter error with optional state term."""
    params = unpack_params(param_vector)

    rhe = RecedingHorizonEstimator(
        dt=scenario.dt,
        N=horizon_n,
        v_thresh=v_thresh,
        w_thresh=w_thresh,
        **params,
    )

    total_steps = scenario.N_total
    U = scenario.U
    Z = scenario.Z
    m_true = scenario.m_true

    window_starts = list(range(0, total_steps - horizon_n + 1, step_size))

    assert param_weight >= 0 and param_weight <= 1, "param_weight must be in [0, 1]"
    state_weight = 1.0 - param_weight

    include_state = state_weight > 0

    total_param_error = 0.0
    total_param_windows = 0.0
    total_state_error = 0.0
    total_state_windows = 0.0

    for start in window_starts:
        end = start + horizon_n
        U_window = U[start:end]
        Z_window = Z[start:end]

        X_opt, m_opt, w_mu, w_nu, _, _ = rhe.estimate(
            U_window, Z_window, step_size=step_size
        )

        mu_true = np.mean(m_true[start:end, 0])
        nu_true = np.mean(m_true[start:end, 1])

        mu_err = abs(m_opt[0] - mu_true)
        nu_err = abs(m_opt[1] - nu_true)

        # Weight errors by confidence: well-excited windows matter more
        # Add small baseline (0.1) so low-excitation windows still contribute
        weight_mu = max(0.1, w_mu)
        weight_nu = max(0.1, w_nu)

        param_loss = 0.5 * (weight_mu * mu_err + weight_nu * nu_err)
        total_param_error += param_loss
        total_param_windows += 1

        if include_state:
            X_true_window = scenario.X_true[start:end]
            # shape (N,3)
            state_rmse = float(np.sqrt(np.mean((X_opt - X_true_window) ** 2)))
            total_state_error += state_rmse
            total_state_windows += 1

    if total_param_windows == 0:
        raise ValueError(
            "No windows were evaluated for parameter error. Check scenario length and horizon size."
        )

    param_term = total_param_error / total_param_windows

    if not include_state or total_state_windows == 0:
        # mild prior: quadratic penalty in log-space toward current baseline
        prior_ref = pack_current_params()
        x = np.asarray(param_vector).ravel()
        eps = 1e-8
        prior_weight = 1e-2
        prior_penalty = prior_weight * float(
            np.sum((np.log(x + eps) - np.log(prior_ref + eps)) ** 2)
        )
        return param_term + prior_penalty

    state_term = total_state_error / total_state_windows

    prior_ref = pack_current_params()
    x = np.asarray(param_vector).ravel()
    eps = 1e-8
    prior_weight = 1e-3
    prior_penalty = prior_weight * float(
        np.sum((np.log(x + eps) - np.log(prior_ref + eps)) ** 2)
    )

    return param_weight * param_term + state_weight * state_term + prior_penalty


def optimize_hyperparameters(
    popsize: int = 15,
    maxiter: int = 50,
    tol: float = 1e-4,
    seed: int = 42,
    workers: int = 1,
    scenario_seed: int = 42,
    noise_sigma: float = 0.01,
    param_weight: float = 0.7,
) -> Tuple[np.ndarray, float]:
    """Run differential evolution to find optimal hyperparameters.

    Args:
        popsize: population size multiplier for DE
        maxiter: maximum DE iterations
        tol: convergence tolerance
        seed: DE random seed
        workers: parallel workers (-1 = all cores)
        scenario_seed: seed for synthetic scenario
        noise_sigma: measurement noise level

    Returns:
        (optimal_params, best_cost)
    """
    scenario = generate_realistic_scenario(
        dt=dt,
        noise_sigma=noise_sigma,
        seed=scenario_seed,
    )

    print(f"Scenario: {scenario.N_total} steps, dt={scenario.dt}")
    print("Terrain segments: good(0-300) → degraded(300-550) → recovered(550-800)")
    print("Angular excitation: ~30% of timesteps + sustained turn sections")
    print(f"Noise σ = {noise_sigma}")
    print()

    # Evaluate current params as baseline
    current_params = pack_current_params()
    current_cost = objective(
        current_params,
        scenario,
        N,
        STEP_SIZE,
        param_weight=param_weight,
    )
    print(f"Current hyperparameters cost: {current_cost:.6f}")
    print(f"Current params: {dict(zip(PARAM_NAMES, current_params))}")
    print()

    iteration_count = [0]
    best_so_far = [current_cost]
    t0 = time.time()

    def callback(xk, convergence):
        iteration_count[0] += 1
        cost = objective(
            xk,
            scenario,
            N,
            STEP_SIZE,
            param_weight=param_weight,
        )
        if cost < best_so_far[0]:
            best_so_far[0] = cost
        elapsed = time.time() - t0
        print(
            f"  Iter {iteration_count[0]:3d} | cost={cost:.6f} | best={best_so_far[0]:.6f} | elapsed={elapsed:.0f}s"
        )

    print(f"Starting differential evolution (popsize={popsize}, maxiter={maxiter})...")
    print()

    result = differential_evolution(
        objective,
        bounds=PARAM_BOUNDS,
        args=(scenario, N, STEP_SIZE, param_weight),
        popsize=popsize,
        maxiter=maxiter,
        tol=tol,
        seed=seed,
        workers=workers,
        callback=callback,
        disp=False,
        polish=False,
    )

    elapsed = time.time() - t0
    print()
    print(f"Optimization complete in {elapsed:.1f}s")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Best cost: {result.fun:.6f}")
    print(
        f"  Improvement: {current_cost:.6f} → {result.fun:.6f} ({(1 - result.fun / current_cost) * 100:.1f}%)"
    )

    return result.x, result.fun


def format_results(optimal_params: np.ndarray, best_cost: float) -> str:
    """Format optimal parameters as a hyperparameters.py replacement."""
    params = unpack_params(optimal_params)

    from data_labeling_pipeline.hyperparameters import v_thresh as cur_vt
    from data_labeling_pipeline.hyperparameters import w_thresh as cur_wt

    lines = []
    lines.append(
        "# Optimized hyperparameters (generated by optimize_hyperparameters.py)"
    )
    lines.append(f"# Best cost: {best_cost:.6f}")
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")
    lines.append(
        f"P_x = np.diag([{params['P_x'][0, 0]:.4f}, {params['P_x'][1, 1]:.4f}, {params['P_x'][2, 2]:.4f}])"
    )
    lines.append(
        f"P_m = np.diag([{params['P_m'][0, 0]:.4f}, {params['P_m'][1, 1]:.4f}, {params['P_m'][2, 2]:.4f}])"
    )
    lines.append(
        f"P_w = np.diag([{params['P_w'][0, 0]:.4f}, {params['P_w'][1, 1]:.4f}, {params['P_w'][2, 2]:.4f}])"
    )
    lines.append("")
    lines.append("N = 10")
    lines.append("STEP_SIZE = max(1, N - 1)")
    lines.append("dt = 0.1")
    lines.append("")
    lines.append(f"v_thresh = {cur_vt:.4f}")
    lines.append(f"w_thresh = {cur_wt:.4f}")
    lines.append(f"max_tie_strengh = {params['max_tie_strengh']:.4f}")

    return "\n".join(lines)


def save_results(optimal_params: np.ndarray, best_cost: float, output_dir: Path):
    """Save optimization results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    params = unpack_params(optimal_params)
    result_dict = {
        "best_cost": best_cost,
        "P_x_diag": params["P_x"].diagonal().tolist(),
        "P_m_diag": params["P_m"].diagonal().tolist(),
        "P_w_diag": params["P_w"].diagonal().tolist(),
        "v_thresh": float(v_thresh),
        "w_thresh": float(w_thresh),
        "max_tie_strengh": params["max_tie_strengh"],
        "current_params": pack_current_params().tolist(),
    }

    json_path = output_dir / "optimized_hyperparameters.json"
    with open(json_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    # Save as Python
    py_path = output_dir / "optimized_hyperparameters.py"
    with open(py_path, "w") as f:
        f.write(format_results(optimal_params, best_cost))
    print(f"Python saved: {py_path}")

    # Print comparison table
    current = pack_current_params()
    print("\nParameter comparison:")
    print(f"{'Parameter':<15} {'Current':>10} {'Optimized':>10} {'Change':>10}")
    print("-" * 50)
    for name, cur, opt in zip(PARAM_NAMES, current, optimal_params):
        change = (opt - cur) / cur * 100 if cur != 0 else float("inf")
        print(f"{name:<15} {cur:>10.4f} {opt:>10.4f} {change:>+9.1f}%")


def main():
    output_dir = Path(__file__).resolve().parent / "benchmark_results"

    print("=" * 60)
    print("  RHE HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print()

    optimal_params, best_cost = optimize_hyperparameters(
        popsize=20,
        maxiter=50,
        tol=1e-4,
        workers=32,  # -1 for all cores
        param_weight=0.85,  # more emphasis on parameter error than state error
    )

    save_results(optimal_params, best_cost, output_dir)

    print("\n" + "=" * 60)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
