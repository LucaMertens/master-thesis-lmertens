"""Run synthetic benchmark scenarios and compute metrics."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from data_labeling_pipeline.hyperparameters import N, STEP_SIZE, P_m, P_w, P_x
from data_labeling_pipeline.rhe import RecedingHorizonEstimator

from .synthetic_data import SyntheticScenario, get_all_scenarios


@dataclass
class ScenarioResult:
    scenario: SyntheticScenario
    mu_estimated: np.ndarray
    nu_estimated: np.ndarray
    mu_true_per_window: np.ndarray
    nu_true_per_window: np.ndarray
    w_mu: np.ndarray  # confidence scores
    w_nu: np.ndarray
    window_centers: np.ndarray  # time indices of window centers
    metrics: Dict = field(default_factory=dict)


def _compute_metrics(
    estimated: np.ndarray, true_vals: np.ndarray, convergence_threshold: float = 0.05
) -> Dict:
    errors = estimated - true_vals
    abs_errors = np.abs(errors)

    metrics = {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error": float(np.max(abs_errors)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
    }

    # Convergence: first window starting a run of 3 consecutive windows below threshold
    converged_mask = abs_errors < convergence_threshold
    if np.any(converged_mask):
        conv_window = None
        consecutive = 0
        for i, c in enumerate(converged_mask):
            if c:
                consecutive += 1
                if consecutive >= 3:
                    conv_window = i - 2  # start of the 3-window run
                    break
            else:
                consecutive = 0
        metrics["convergence_window"] = conv_window
    else:
        metrics["convergence_window"] = None

    return metrics


def run_scenario(
    scenario: SyntheticScenario, horizon_n: int = None, step_size: int = None
) -> ScenarioResult:
    if horizon_n is None:
        horizon_n = N
    if step_size is None:
        step_size = STEP_SIZE

    rhe = RecedingHorizonEstimator(
        dt=scenario.dt,
        N=horizon_n,
        P_x=P_x,
        P_m=P_m,
        P_w=P_w,
    )

    total_steps = scenario.N_total
    U = scenario.U
    Z = scenario.Z
    m_true = scenario.m_true

    window_starts = list(range(0, total_steps - horizon_n + 1, step_size))
    num_windows = len(window_starts)

    mu_est = np.zeros(num_windows)
    nu_est = np.zeros(num_windows)
    mu_true_w = np.zeros(num_windows)
    nu_true_w = np.zeros(num_windows)
    w_mu_arr = np.zeros(num_windows)
    w_nu_arr = np.zeros(num_windows)
    window_centers = np.zeros(num_windows)

    for idx, start in enumerate(window_starts):
        end = start + horizon_n
        U_window = U[start:end]
        Z_window = Z[start:end]

        X_opt, m_opt, w_mu, w_nu, v_mean, omega_mean = rhe.estimate(
            U_window, Z_window, step_size=step_size
        )

        mu_est[idx] = m_opt[0]
        nu_est[idx] = m_opt[1]
        w_mu_arr[idx] = w_mu
        w_nu_arr[idx] = w_nu

        # True parameters averaged over the window
        mu_true_w[idx] = np.mean(m_true[start:end, 0])
        nu_true_w[idx] = np.mean(m_true[start:end, 1])

        window_centers[idx] = (start + end) / 2.0

    metrics = {
        "mu": _compute_metrics(mu_est, mu_true_w),
        "nu": _compute_metrics(nu_est, nu_true_w),
        "mean_w_mu": float(np.mean(w_mu_arr)),
        "mean_w_nu": float(np.mean(w_nu_arr)),
    }

    return ScenarioResult(
        scenario=scenario,
        mu_estimated=mu_est,
        nu_estimated=nu_est,
        mu_true_per_window=mu_true_w,
        nu_true_per_window=nu_true_w,
        w_mu=w_mu_arr,
        w_nu=w_nu_arr,
        window_centers=window_centers,
        metrics=metrics,
    )


def run_all_scenarios() -> List[ScenarioResult]:
    scenarios = get_all_scenarios()
    results = []

    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(scenarios)}] Running: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"  Steps: {scenario.N_total}, dt: {scenario.dt}, noise: {scenario.noise_sigma}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            result = run_scenario(scenario)
            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.1f}s")
            print(f"  μ: MAE={result.metrics['mu']['mae']:.4f}, RMSE={result.metrics['mu']['rmse']:.4f}")
            print(f"  ν: MAE={result.metrics['nu']['mae']:.4f}, RMSE={result.metrics['nu']['rmse']:.4f}")
            print(f"  Confidence: w_μ={result.metrics['mean_w_mu']:.3f}, w_ν={result.metrics['mean_w_nu']:.3f}")
            results.append(result)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.1f}s: {e}")
            results.append(None)

    return results


def print_summary_table(results: List[ScenarioResult]):
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    header = f"{'Scenario':<30} {'MAE(μ)':>8} {'RMSE(μ)':>9} {'MAE(ν)':>8} {'RMSE(ν)':>9} {'w_μ':>6} {'w_ν':>6} {'Conv(μ)':>8} {'Conv(ν)':>8}"
    print(header)
    print("-" * 100)

    for r in results:
        if r is None:
            print(f"{'FAILED':<30}")
            continue

        conv_mu = r.metrics["mu"].get("convergence_window")
        conv_nu = r.metrics["nu"].get("convergence_window")
        conv_mu_str = f"{conv_mu}" if conv_mu is not None else "N/A"
        conv_nu_str = f"{conv_nu}" if conv_nu is not None else "N/A"

        row = (
            f"{r.scenario.name:<30} "
            f"{r.metrics['mu']['mae']:>8.4f} "
            f"{r.metrics['mu']['rmse']:>9.4f} "
            f"{r.metrics['nu']['mae']:>8.4f} "
            f"{r.metrics['nu']['rmse']:>9.4f} "
            f"{r.metrics['mean_w_mu']:>6.3f} "
            f"{r.metrics['mean_w_nu']:>6.3f} "
            f"{conv_mu_str:>8} "
            f"{conv_nu_str:>8}"
        )
        print(row)

    print("=" * 100)


def save_metrics(results: List[ScenarioResult], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict = {}
    for r in results:
        if r is None:
            continue
        m = {
            "description": r.scenario.description,
            "noise_sigma": r.scenario.noise_sigma,
            "mu": r.metrics["mu"],
            "nu": r.metrics["nu"],
            "mean_w_mu": r.metrics["mean_w_mu"],
            "mean_w_nu": r.metrics["mean_w_nu"],
        }
        metrics_dict[r.scenario.name] = m

    path = output_dir / "benchmark_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    print(f"\nMetrics saved to {path}")
