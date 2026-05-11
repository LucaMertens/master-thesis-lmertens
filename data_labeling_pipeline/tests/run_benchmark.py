#!/usr/bin/env python3
"""
Run benchmark scenarios and generate reports.

Usage:
uv run python -m tests.run_benchmark
"""

import sys
from pathlib import Path

# Add parent dir to path so we can import the tests package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.rhe_benchmark import print_summary_table, run_all_scenarios, save_metrics
from tests.rhe_benchmark_plots import generate_all_plots


def main():
    output_dir = Path(__file__).resolve().parent / "benchmark_results"

    print("=" * 60)
    print("  RHE DATA LABELING PIPELINE BENCHMARK")
    print("=" * 60)

    results = run_all_scenarios()
    print_summary_table(results)
    save_metrics(results, output_dir)
    generate_all_plots(results, output_dir)

    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print(f"  Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
