#!/usr/bin/env python
"""Overlay WayFASTER run trajectories, which already share a world frame."""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared_utils.paths import DATALOCAL_ROOT

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from align_runs import RUNS_2023_06_28, load_run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", default="zed2")
    p.add_argument("--out", type=pathlib.Path, default=DATALOCAL_ROOT / "wayfaster_split_videos" / "paths.png")
    args = p.parse_args()

    runs = [load_run(args.cam, s, f) for s, f in RUNS_2023_06_28]
    cmap = plt.get_cmap("tab10")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 8))

    for k, r in enumerate(runs):
        ax0.plot(r["xy"][:, 0], r["xy"][:, 1], lw=1.1, color=cmap(k), label=f"{r['name']} ({r['split']})")
        ax0.plot(*r["xy"][0], "o", color=cmap(k), ms=6, mec="k", mew=0.6)
    ax0.set_title("All 9 runs, raw odometry coordinates")
    ax0.legend(fontsize=7, loc="best")

    for r in runs:
        c = "tab:blue" if r["split"] == "train" else "tab:orange"
        ax1.plot(r["xy"][:, 0], r["xy"][:, 1], lw=1.1, color=c, alpha=0.75)
    ax1.plot([], [], color="tab:blue", label="train")
    ax1.plot([], [], color="tab:orange", label="valid")
    ax1.set_title("Coloured by split")
    ax1.legend(fontsize=9)

    for a in (ax0, ax1):
        a.set_aspect("equal")
        a.grid(alpha=0.3)
        a.set_xlabel("x [m]")
        a.set_ylabel("y [m]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
