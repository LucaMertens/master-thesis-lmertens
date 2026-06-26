"""Estimate, per epoch and without training, how OVERSAMPLE_FRACTION and the LDS
exponent alpha reshape the mu loss: fraction moves cell *exposure*, alpha moves
*loss-mass share* and the max/min weight ratio (the instability indicator).

Usage:
    uv run --project training/traversability_network python \
        training/traversability_network/tests/oversampling_effect.py
"""

import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from traversability_network.config import get_cfg
from traversability_network.data.grandtour_dataset import GrandTourDataset

CFG = Path(__file__).resolve().parents[1] / "configs" / "grandtour.yaml"
NBINS = 5  # 0.2-wide mu bins, matching prepare_weights
LOW = 3  # bins 0,1,2 cover mu < 0.6
ALPHAS = [0.25, 0.5, 0.75, 1.0]
FRACTIONS = [0.3, 0.4, 0.5, 0.7]
BIN_LABELS = [f"[{i / NBINS:.1f},{(i + 1) / NBINS:.1f})" for i in range(NBINS)]


def all_missions(cfg):
    """Every mission with the rhe labels and extracted manifests this script needs."""
    need_lbl = ["rhe/rhe_timestamp.npy", "rhe/rhe_trav_labels.npy",
                "rhe/rhe_mu_observability.npy"]
    need_ext = ["image_manifest.npz", "depth_manifest.npz", "cam2base.npy",
                "intrinsics.npy", "dlio_odom.npy"]
    out = []
    for m in sorted(set(os.listdir(cfg.DATASET.DATA_ROOT)) & set(os.listdir(cfg.DATASET.LABELS_ROOT))):
        if all(os.path.exists(os.path.join(cfg.DATASET.LABELS_ROOT, m, f)) for f in need_lbl) and \
           all(os.path.exists(os.path.join(cfg.DATASET.DATA_ROOT, m, f)) for f in need_ext):
            out.append(m)
    return out


def lds_weights(all_mu, alpha):
    """Per-bin LDS weight (1/density)^alpha, smoothed exactly like prepare_weights."""
    values, _ = np.histogram(all_mu, bins=NBINS, range=(0, 1), density=True)
    values = gaussian_filter1d(values, sigma=1.0, mode="constant")
    return (1.0 / values) ** alpha


def bin_exposure(ds):
    """Per global window x mu-bin: observability-weighted cell count over its horizon.

    Returns (NBINS, n_windows). Summed against window multiplicities this gives the
    observability-gated cell exposure per bin, i.e. the alpha-independent loss budget.
    """
    n_flag = min(ds.horizon, 300)  # match _flagged_windows: near-horizon cutoff
    cols = []
    for bag in ds.rosbags:
        nw = len(bag["image_timestamp"]) - ds.n_frames + 1
        st = np.asarray(bag["states_timestamp"])
        mu = np.asarray(bag["traversability"])[:, 0]
        observability = np.asarray(bag["mu_observability"])
        b = np.clip((mu * NBINS).astype(int), 0, NBINS - 1)
        img = np.asarray(bag["image_timestamp"])[:nw]
        lo = np.searchsorted(st, img, "left")
        hi = np.searchsorted(st, img + n_flag * ds.dt, "left")
        per = np.empty((NBINS, nw))
        for k in range(NBINS):
            pref = np.concatenate(([0], np.cumsum(np.where(b == k, observability, 0.0))))
            per[k] = pref[hi] - pref[lo]
        cols.append(per)
    return np.concatenate(cols, axis=1)


def main():
    cfg = get_cfg(str(CFG))
    missions = all_missions(cfg)
    ds = GrandTourDataset(cfg, missions, train=True)

    all_mu = np.concatenate([np.asarray(b["traversability"])[:, 0] for b in ds.rosbags])
    obs_xpo = bin_exposure(ds)  # (NBINS, n_windows)
    flagged = ds._flagged_windows()
    n_win = obs_xpo.shape[1]

    print(f"\nmissions: {len(missions)}  windows: {n_win}")
    print(f"flagged windows (contain a low-mu cell): {flagged.mean():.1%}  "
          f"(natural rate, the oversampler is a no-op below this)")

    # alpha -> per-bin LDS weight and the max/min ratio (frac-independent)
    print("\nLDS weight per mu bin (and max/min ratio):")
    print(f"{'alpha':>6}  " + "".join(f"{lab:>11}" for lab in BIN_LABELS) + f"{'ratio':>9}")
    for a in ALPHAS:
        w = lds_weights(all_mu, a)
        print(f"{a:>6.2f}  " + "".join(f"{x:>11.2f}" for x in w)
              + f"{w.max() / w.min():>9.0f}x")

    # frac -> observability-gated exposure share of low-mu cells (alpha-independent)
    print("\nlow-mu (<0.6) cell exposure share vs oversample_fraction:")
    print(f"{'frac':>6}{'exposure':>10}")
    for frac in [0.0] + FRACTIONS:
        ds.OVERSAMPLE_FRACTION = frac
        counts = np.bincount(np.asarray(ds._oversampled_indices()), minlength=n_win)
        expo = obs_xpo @ counts
        print(f"{frac:>6.2f}{expo[:LOW].sum() / expo.sum():>10.2%}")

    # combined: low-mu loss-mass share over the alpha x frac grid
    print("\nlow-mu (<0.6) share of total mu-loss mass  (rows alpha, cols frac):")
    print(f"{'alpha\\frac':>10}" + "".join(f"{f:>10.2f}" for f in FRACTIONS))
    grid = {}
    for a in ALPHAS:
        w = lds_weights(all_mu, a)
        row = []
        for frac in FRACTIONS:
            ds.OVERSAMPLE_FRACTION = frac
            counts = np.bincount(np.asarray(ds._oversampled_indices()), minlength=n_win)
            mass = w * (obs_xpo @ counts)
            share = mass[:LOW].sum() / mass.sum()
            grid[(a, frac)] = share
            row.append(share)
        print(f"{a:>10.2f}" + "".join(f"{x:>10.2%}" for x in row))

    print(f"\nraw low-mu loss share with no reweighting and no oversampling: "
          f"{obs_xpo[:LOW].sum() / obs_xpo.sum():.2%}")
    print("frac moves exposure (top table); alpha moves loss share + weight ratio.")
    print("High alpha buys low-mu loss share at the cost of a large weight ratio "
          "(gradient-variance risk).")


if __name__ == "__main__":
    main()
