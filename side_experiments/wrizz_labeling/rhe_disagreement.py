"""Human vs RHE disagreement on rhe_overlap pairs, by observability decile and mission.

Run: uv run python side_experiments/wrizz_labeling/rhe_disagreement.py"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import rhe as rhe_mod
from export_labels import merge_amendments


def rows(labels, cfg, deadband):
    """One row per rhe_overlap pair with both endpoints RHE-resolved."""
    cache, out = {}, []
    for r in labels:
        if r["pair_source"] != "rhe_overlap" or r["t"] is None:
            continue
        vals = []
        for side in ("a", "b"):
            e = r[side]
            if e.get("rhe_index") is None:
                break
            if e["mission"] not in cache:
                cache[e["mission"]] = rhe_mod.RheLabels(
                    cfg.DATASET.LABELS_ROOT, e["mission"],
                    os.path.join(cfg.DATASET.DATA_ROOT, e["mission"]))
            m = cache[e["mission"]]
            vals.append((float(m.mu[e["rhe_index"]]), float(m.obs[e["rhe_index"]])))
        if len(vals) != 2:
            continue
        (mu_a, obs_a), (mu_b, obs_b) = vals
        t_rhe = rhe_mod.compare(mu_a, mu_b, deadband)
        out.append(dict(mission=r["a"]["mission"], t_human=r["t"], t_rhe=t_rhe,
                        obs=min(obs_a, obs_b), gap=abs(mu_b - mu_a),
                        mu_min=min(mu_a, mu_b), split=r["split"]))
    return out


def table(name, keys, groups):
    print(f"  by {name}:")
    for key, sub in zip(keys, groups):
        if not len(sub):
            continue
        bad = np.mean([x["t_human"] != x["t_rhe"] for x in sub])
        ties = np.mean([x["t_human"] == 0 for x in sub])
        rhe_ties = np.mean([x["t_rhe"] == 0 for x in sub])
        flips = np.mean([x["t_human"] * x["t_rhe"] == -1 for x in sub])
        print(f"    {key:>14}  n={len(sub):4d}  disagree {bad:.3f}  "
              f"sign-flip {flips:.3f}  human-tie {ties:.3f}  rhe-tie {rhe_ties:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="side_experiments/wrizz_labeling/output")
    ap.add_argument("--cfg", default=common.CFG_FILE)
    ap.add_argument("--deadband", type=float, default=0.25)
    args = ap.parse_args()

    cfg = common.load_cfg(args.cfg)
    labels = merge_amendments(common.read_jsonl(os.path.join(args.out, "labels.jsonl")))
    labels = [r for r in labels
              if not any(common.excluded(r[s]["mission"], r[s]["ts_ns"]) for s in ("a", "b"))]
    data = [x for x in rows(labels, cfg, args.deadband) if x["t_rhe"] is not None]
    n_amb = len(rows(labels, cfg, args.deadband)) - len(data)
    bad = np.mean([x["t_human"] != x["t_rhe"] for x in data])
    print(f"rhe_overlap pairs comparable {len(data)} "
          f"({n_amb} in the ambiguous mu band, dropped), deadband {args.deadband}")
    print(f"overall disagreement {bad:.3f}")

    obs = np.array([x["obs"] for x in data])
    edges = np.quantile(obs, np.linspace(0, 1, 11))
    edges[-1] += 1e-9
    idx = np.clip(np.searchsorted(edges, obs, "right") - 1, 0, 9)
    table("observability decile",
          [f"{edges[i]:.3f}-{edges[i + 1]:.3f}" for i in range(10)],
          [[x for x, j in zip(data, idx) if j == i] for i in range(10)])

    missions = sorted({x["mission"] for x in data})
    table("mission", missions,
          [[x for x in data if x["mission"] == m] for m in missions])

    bands = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    table("lower RHE mu", [f"{lo:.1f}-{hi:.1f}" for lo, hi in bands],
          [[x for x in data if lo <= x["mu_min"] < hi] for lo, hi in bands])


if __name__ == "__main__":
    main()
