"""Plot utilities for benchmark results."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .rhe_benchmark import ScenarioResult

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.dpi"] = 150


def plot_scenario_timeseries(result: ScenarioResult, output_dir: Path):
    """Plot command, estimate, and error time-series for one scenario."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f"Scenario: {result.scenario.name}\n{result.scenario.description}", fontsize=13)

    t_steps = np.arange(result.scenario.N_total) * result.scenario.dt
    t_windows = result.window_centers * result.scenario.dt

    # Panel 1: Commanded inputs
    ax = axes[0]
    ax.plot(t_steps, result.scenario.U[:, 0], label="v_cmd", color="steelblue", linewidth=1.5)
    ax.plot(t_steps, result.scenario.U[:, 1], label="ω_cmd", color="coral", linewidth=1.5)
    ax.set_ylabel("Command")
    ax.legend(loc="upper right")
    ax.set_title("Commanded Inputs")

    # Panel 2: mu - true vs estimated
    ax = axes[1]
    ax.plot(t_windows, result.mu_true_per_window, label="μ_true", color="forestgreen", linewidth=2, linestyle="--")
    ax.plot(t_windows, result.mu_estimated, label="μ_est", color="steelblue", linewidth=1.5)
    ax.fill_between(
        t_windows,
        0, result.w_mu,
        alpha=0.15, color="steelblue", label="w_μ (confidence)"
    )
    ax.set_ylabel("μ (linear trav.)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right")
    ax.set_title("Linear Traversability (μ)")

    # Panel 3: nu - true vs estimated
    ax = axes[2]
    ax.plot(t_windows, result.nu_true_per_window, label="ν_true", color="forestgreen", linewidth=2, linestyle="--")
    ax.plot(t_windows, result.nu_estimated, label="ν_est", color="coral", linewidth=1.5)
    ax.fill_between(
        t_windows,
        0, result.w_nu,
        alpha=0.15, color="coral", label="w_ν (confidence)"
    )
    ax.set_ylabel("ν (angular trav.)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right")
    ax.set_title("Angular Traversability (ν)")

    # Panel 4: Absolute errors
    ax = axes[3]
    mu_err = np.abs(result.mu_estimated - result.mu_true_per_window)
    nu_err = np.abs(result.nu_estimated - result.nu_true_per_window)
    ax.plot(t_windows, mu_err, label="|μ_err|", color="steelblue", linewidth=1.5)
    ax.plot(t_windows, nu_err, label="|ν_err|", color="coral", linewidth=1.5)
    ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.7, label="convergence threshold")
    ax.set_ylabel("Absolute Error")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.set_title("Estimation Error")

    plt.tight_layout()
    path = output_dir / f"timeseries_{result.scenario.name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_scenario_scatter(result: ScenarioResult, output_dir: Path):
    """Scatter plot: estimated vs true values for mu and nu."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Scatter: {result.scenario.name}", fontsize=13)

    for ax, est, true, name, color in [
        (axes[0], result.mu_estimated, result.mu_true_per_window, "μ", "steelblue"),
        (axes[1], result.nu_estimated, result.nu_true_per_window, "ν", "coral"),
    ]:
        ax.scatter(true, est, alpha=0.5, s=20, color=color, edgecolors="none")
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="ideal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"{name}_true")
        ax.set_ylabel(f"{name}_estimated")
        ax.set_title(f"{name}: Estimated vs True")
        ax.legend()
        ax.set_aspect("equal")

    plt.tight_layout()
    path = output_dir / f"scatter_{result.scenario.name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_scenario_error_distribution(result: ScenarioResult, output_dir: Path):
    """Histogram of estimation errors for mu and nu."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Error Distribution: {result.scenario.name}", fontsize=13)

    for ax, est, true, name, color in [
        (axes[0], result.mu_estimated, result.mu_true_per_window, "μ", "steelblue"),
        (axes[1], result.nu_estimated, result.nu_true_per_window, "ν", "coral"),
    ]:
        errors = est - true
        ax.hist(errors, bins=30, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
        ax.axvline(x=np.mean(errors), color="red", linestyle="-", alpha=0.7, label=f"mean={np.mean(errors):.4f}")
        ax.set_xlabel(f"{name} error")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}: Error Distribution")
        ax.legend()

    plt.tight_layout()
    path = output_dir / f"error_dist_{result.scenario.name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_bar_chart(results: List[ScenarioResult], output_dir: Path):
    """Bar chart comparing MAE and RMSE across all scenarios."""
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return

    names = [r.scenario.name for r in valid_results]
    mae_mu = [r.metrics["mu"]["mae"] for r in valid_results]
    rmse_mu = [r.metrics["mu"]["rmse"] for r in valid_results]
    mae_nu = [r.metrics["nu"]["mae"] for r in valid_results]
    rmse_nu = [r.metrics["nu"]["rmse"] for r in valid_results]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.bar(x - 1.5 * width, mae_mu, width, label="MAE(μ)", color="steelblue", alpha=0.8)
    ax.bar(x - 0.5 * width, rmse_mu, width, label="RMSE(μ)", color="steelblue", alpha=0.5)
    ax.bar(x + 0.5 * width, mae_nu, width, label="MAE(ν)", color="coral", alpha=0.8)
    ax.bar(x + 1.5 * width, rmse_nu, width, label="RMSE(ν)", color="coral", alpha=0.5)

    ax.set_ylabel("Error")
    ax.set_title("Benchmark Summary: MAE and RMSE by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.7, label="target threshold")

    plt.tight_layout()
    path = output_dir / "summary_bar_chart.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_noise_robustness(results: List[ScenarioResult], output_dir: Path):
    """Plot MAE vs noise level for the noise scenarios."""
    noise_results = [r for r in results if r is not None and r.scenario.name.startswith("noise_sigma_")]
    if not noise_results:
        return

    noise_results.sort(key=lambda r: r.scenario.noise_sigma)

    sigmas = [r.scenario.noise_sigma for r in noise_results]
    mae_mu = [r.metrics["mu"]["mae"] for r in noise_results]
    mae_nu = [r.metrics["nu"]["mae"] for r in noise_results]
    rmse_mu = [r.metrics["mu"]["rmse"] for r in noise_results]
    rmse_nu = [r.metrics["nu"]["rmse"] for r in noise_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Noise Robustness", fontsize=13)

    # MAE plot
    ax = axes[0]
    ax.plot(sigmas, mae_mu, "o-", label="MAE(μ)", color="steelblue", linewidth=2, markersize=8)
    ax.plot(sigmas, mae_nu, "s-", label="MAE(ν)", color="coral", linewidth=2, markersize=8)
    ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("MAE")
    ax.set_title("MAE vs Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE plot
    ax = axes[1]
    ax.plot(sigmas, rmse_mu, "o-", label="RMSE(μ)", color="steelblue", linewidth=2, markersize=8)
    ax.plot(sigmas, rmse_nu, "s-", label="RMSE(ν)", color="coral", linewidth=2, markersize=8)
    ax.axhline(y=0.05, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "noise_robustness.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confidence_tracking(results: List[ScenarioResult], output_dir: Path):
    """Plot confidence score evolution for key scenarios."""
    key_names = [
        "constant_serpentine",
        "straight_walking",
        "turning_in_place",
        "full_stop",
        "mixed_excitation",
        "step_ramp",
    ]
    key_results = [r for r in results if r is not None and r.scenario.name in key_names]
    if not key_results:
        return

    fig, axes = plt.subplots(len(key_results), 1, figsize=(14, 3.5 * len(key_results)))
    if len(key_results) == 1:
        axes = [axes]

    fig.suptitle("Confidence Score Tracking Across Scenarios", fontsize=13)

    for ax, result in zip(axes, key_results):
        t_windows = result.window_centers * result.scenario.dt
        ax.plot(t_windows, result.w_mu, label="w_μ", color="steelblue", linewidth=1.5)
        ax.plot(t_windows, result.w_nu, label="w_ν", color="coral", linewidth=1.5)
        ax.set_ylabel("Confidence")
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(result.scenario.name)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    path = output_dir / "confidence_tracking.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_all_plots(results: List[ScenarioResult], output_dir: Path):
    """Generate all benchmark plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating per-scenario plots...")
    for result in results:
        if result is None:
            continue
        print(f"\n  Plotting: {result.scenario.name}")
        plot_scenario_timeseries(result, output_dir)
        plot_scenario_scatter(result, output_dir)
        plot_scenario_error_distribution(result, output_dir)

    print("\nGenerating summary plots...")
    plot_summary_bar_chart(results, output_dir)
    plot_noise_robustness(results, output_dir)
    plot_confidence_tracking(results, output_dir)

    print(f"\nAll plots saved to: {output_dir}")
