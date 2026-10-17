"""Score a checkpoint on both human HDR and RHE auroc_low in one forward pass.

Run: uv run python -m traversability_network.adaptations.score_human_hdr \
        --cfg_file <data config> --checkpoint <ckpt> [--model_cfg <config the ckpt was built with>]
"""

import argparse

import torch
from torch.utils.data import DataLoader

from ..config import get_cfg
from ..network.trainer import TrainingModule
from . import human_pairs as HP
from .grandtour_dataset import GrandTourDataset
from .metrics import RISKY_THRESHOLD, auroc_from_hist, score_range
from .rizz import HEADLINE_TAU, balanced_hdr, hdr_counts


def load_model(model_cfg, checkpoint, device):
    """Load weights into a module built from the checkpoint's config; asserts on prefix mismatch."""
    model_cfg.TRAINING.COMPILE = False
    module = TrainingModule(model_cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    state = {k.replace("model._orig_mod.", "model."): v for k, v in state.items()}
    missing, unexpected = module.load_state_dict(state, strict=False)
    hit = len(module.state_dict()) - len(missing)
    assert hit > 0.9 * len(module.state_dict()), (
        f"only {hit}/{len(module.state_dict())} parameters matched the checkpoint "
        f"(missing {missing[:5]}..., unexpected {unexpected[:5]}...)")
    return module.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_file", required=True, help="config defining the data and the human index")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_cfg", default=None, help="config the checkpoint's model was built with")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--taus", default="0.02,0.05,0.1,0.15,0.25,0.35,0.5,0.75",
                    help="thresholds to report HDR at")
    ap.add_argument("--split", default="valid", choices=("valid", "train"),
                    help="which missions' pairs to score")
    args = ap.parse_args()

    cfg = get_cfg(args.cfg_file)
    cfg.TRAINING.HUMAN_WEIGHT = max(cfg.TRAINING.HUMAN_WEIGHT, 1.0)  # build the schedule
    train = GrandTourDataset(cfg, cfg.DATASET.TRAIN_MISSIONS, train=True)
    valid = GrandTourDataset(cfg, cfg.DATASET.VALID_MISSIONS, weights=train.weights)
    HP.attach_human_pairs(cfg, train, valid)
    table = valid.human.table
    if args.split == "train":
        # same per-annotation scoring as valid, on the trained pairs
        valid = GrandTourDataset(cfg, cfg.DATASET.TRAIN_MISSIONS, weights=train.weights)
        valid.human = HP.HumanPairs(table, valid, cfg.MODEL.GRID_BOUNDS, k=1,
                                    ineq_fraction=0.0, trainable_only=True)
        print(valid.human.summary("train-as-eval"))

    module = load_model(get_cfg(args.model_cfg or args.cfg_file), args.checkpoint, args.device)
    taus = tuple(float(x) for x in args.taus.split(","))
    metric = HP.HumanHdrMetrics(table, taus=taus).to(args.device)
    scores = torch.zeros(2, 256, device=args.device)

    loader = DataLoader(valid, batch_size=8, num_workers=0, drop_last=True)
    with torch.no_grad():
        for batch in loader:
            color, voxel, k_nat, k, ext, path, target, weights, dtarget, dmask, _, human = batch
            def to(t):
                return t.to(args.device)
            trav = module.model(to(color), to(voxel), to(k_nat), to(k), to(ext), to(dtarget))[0]
            metric.update(trav, {key: to(v) for key, v in human.items()}, cfg.MODEL.GRID_BOUNDS)
            # RHE side: same gated cells as metrics.MuBinMetrics auroc_low
            pred, mask = module._sample_path(to(path), trav)
            keep = mask.bool() & (to(weights)[..., 0] > 0)
            pred_mu, true_mu = pred[..., 0][keep], to(target)[..., 0][keep]
            sb = (pred_mu * 256).long().clamp_(0, 255)
            low = true_mu < RISKY_THRESHOLD
            ones = torch.ones_like(sb, dtype=scores.dtype)
            scores[0].index_add_(0, sb[low], ones[low])
            scores[1].index_add_(0, sb[~low], ones[~low])

    have = ((metric.cnt > 0).all(-1) & metric.eligible.to(metric.cnt.device)).nonzero().squeeze(1)
    p = metric.acc[have] / metric.cnt[have]
    counts = hdr_counts(p[:, 0], p[:, 1], metric.t[have], metric.taus)
    print(f"\n{args.checkpoint}  [{args.split} pairs, n={have.numel()}]")
    print(f"  {'tau':>6} {'balanced':>9} {'all':>7} {'eq':>7} {'ineq':>7}")
    for i, tau in enumerate(metric.taus.tolist()):
        row = [counts[i, j, 0] / counts[i, j, 1] if counts[i, j, 1] > 0 else float("nan")
               for j in range(3)]
        mark = "  <- headline" if abs(tau - HEADLINE_TAU) < 1e-9 else ""
        print(f"  {tau:6.2f} {balanced_hdr(counts[i]).item():9.3f} "
              f"{row[0]:7.3f} {row[1]:7.3f} {row[2]:7.3f}{mark}")
    # distance between predicted endpoints, which a tau sweep reads
    d = (p[:, 1] - p[:, 0]).abs()
    t = metric.t[have]
    for name, sel in (("tie", t == 0), ("inequality", t != 0)):
        if sel.any():
            q = d[sel].quantile(torch.tensor([0.1, 0.5, 0.9], device=d.device))
            print(f"  |p_b - p_a| on {name:10s} pairs: p10 {q[0]:.3f} "
                  f"median {q[1]:.3f} p90 {q[2]:.3f}")
    auroc = auroc_from_hist(scores)
    rng = score_range(scores)
    print(f"  auroc_low (RHE, higher better)                      "
          f"{auroc:.3f}" if auroc is not None else "  auroc_low  n/a")
    if rng is not None:
        print(f"    pred p01 {rng[0]:.3f}  p50 {rng[1]:.3f}  frac_low {rng[2]:.3f}")


if __name__ == "__main__":
    main()
