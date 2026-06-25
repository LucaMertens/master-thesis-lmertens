"""Plot the observability score vs mean command magnitude.

uv run --project preprocessing/data_labeling_pipeline python documentation/scripts/plot_observability.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from data_labeling_pipeline.hyperparameters import obs_hi, obs_lo
from data_labeling_pipeline.rhe import observability

OUT = Path(__file__).resolve().parents[1] / "figures" / "observability.pdf"


def main():
    x = np.linspace(0, 1.3 * obs_hi, 400)
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(x, observability(x), color="steelblue", linewidth=2)
    ax.axvline(obs_lo, color="gray", linestyle="--", linewidth=1)
    ax.axvline(obs_hi, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(r"mean command magnitude  $\bar v$ [m/s]  or  $\bar\omega$ [rad/s]")
    ax.set_ylabel(r"observability  $o_\mu$ or $o_\nu$")
    ax.set_xlim(0, x[-1])
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
