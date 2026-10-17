"""Compare a run's in-loop per-mission metrics against a full offline pass of the same checkpoint.

They should agree except for AUROC score-bin ties.

    uv run python -m traversability_network.adaptations.verify_mission_metrics \
        --run-id <wandb id> [--checkpoint best] [--device cuda:0]
"""

import argparse
import os

import numpy as np

from . import mission_metrics
from .extract_mu_predictions import load_from_checkpoint, run_mission
from .render_mission_video import resolve_checkpoint

PROJECT = "master-thesis-lmertens"
ENTITY = os.environ.get("WANDB_ENTITY")


def offline(checkpoint, device):
    configs, model = load_from_checkpoint(checkpoint, device)
    ranges = getattr(configs.DATASET, "VALID_TIME_RANGES", None) or {}
    pred, true, mission = [], [], []
    for name in configs.DATASET.VALID_MISSIONS:
        res = run_mission(configs, model, name, device, 8, 8, 1, False, ranges)
        if res is None:
            continue
        pred.append(res["pred_mu"])
        true.append(res["true_mu"])
        mission.append(np.full(len(res["pred_mu"]), name))
    return mission_metrics.by_mission(
        np.concatenate(pred), np.concatenate(true), np.concatenate(mission)
    )


def logged(run_id, step):
    """The valid_mission/* values the run logged at ``step``, or {} if unavailable."""
    try:
        import wandb

        api = wandb.Api()
        path = f"{ENTITY}/{PROJECT}/{run_id}" if ENTITY else f"{PROJECT}/{run_id}"
        run = api.run(path)
        keys = [f"valid_mission/{k}" for k in mission_metrics_keys()]
        rows = run.history(keys=keys + ["trainer/global_step"], pandas=False)
        near = min(rows, key=lambda r: abs(r.get("trainer/global_step", -1) - step))
        return {k: near[k] for k in keys if k in near}
    except Exception as exc:
        print(f"wandb history unavailable ({exc})")
        return {}


def mission_metrics_keys():
    return ("calib_slope", "auroc_risky", "recall_pm1", "false_alarm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint", default="best", help='"best" or an archived step')
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch

    path = resolve_checkpoint(args.run_id, args.checkpoint)
    step = int(path.split("step=")[1].split(".")[0])
    print(f"scoring {os.path.basename(path)} (step {step})", flush=True)
    per, mean = offline(path, torch.device(args.device))
    for name, s in per.items():
        print(f"  {name:8s} " + " ".join(f"{k}={s[k]:.3f}" for k in mission_metrics_keys())
              + f" n_risky={int(s['n_risky'])}")
    print(f"  {'mean':8s} " + " ".join(f"{k}={mean[k]:.3f}" for k in mission_metrics_keys())
          + f" ({mean['n_missions_scored']}/{mean['n_missions']} missions scored)")

    live = logged(args.run_id, step)
    for key in mission_metrics_keys():
        full = f"valid_mission/{key}"
        if full in live and live[full] is not None:
            print(f"  {key:12s} offline {mean[key]:.4f}  in-loop {live[full]:.4f}  "
                  f"diff {abs(mean[key] - live[full]):.4f}")


if __name__ == "__main__":
    main()
