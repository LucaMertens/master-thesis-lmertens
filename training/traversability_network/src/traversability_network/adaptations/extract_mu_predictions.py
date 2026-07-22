"""Dumps per-cell (pred_mu, true_mu) pairs from a checkpoint for the 3-class probe (side_experiments/dino_mu_probe/run_3class.py)."""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .grandtour_dataset import GrandTourDataset
from ..network.trainer import TrainingModule


def load_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    configs = ckpt["hyper_parameters"]["configs"]
    configs.TRAINING.COMPILE = False
    model = TrainingModule(configs)
    state_dict = {k.replace("._orig_mod", ""): v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"non-strict load: missing={missing} unexpected={unexpected}")
    return configs, model.eval().to(device)


@torch.no_grad()
def run_mission(configs, model, mission, device, batch_size, workers, stride, dump_features):
    ds = GrandTourDataset(configs, [mission], train=False)
    if len(ds) == 0:
        return None
    if stride > 1:
        ds = Subset(ds, range(0, len(ds), stride))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    bev = {}
    hook = model.model.decoder.register_forward_hook(lambda m, i, o: bev.update(f=o)) if dump_features else None
    is_class3 = getattr(configs.TRAINING, "LOSS_TYPE", None) == "class3"

    out = {"pred_mu": [], "true_mu": [], "pred_class": [], "bev_feat": []}
    for batch in loader:
        color, depth_voxel, K_full, K, ext, states, target_trav, trav_weights, depth_target, _, *extra = (
            x.to(device) for x in batch
        )
        image_feats = extra[0] if extra else None
        trav_map, mu_logits, _, _ = model.model(color, depth_voxel, K_full, K, ext, depth_target, image_feats=image_feats)
        pred, mask = model._sample_path(states, trav_map)
        mask = mask.bool() & (trav_weights[..., 0] > 0)
        out["pred_mu"].append(pred[..., 0][mask].cpu().numpy())
        out["true_mu"].append(target_trav[..., 0][mask].cpu().numpy())
        if is_class3:
            logits, _ = model._sample_path(states, mu_logits)
            out["pred_class"].append(logits.argmax(-1)[mask].cpu().numpy().astype(np.int8))
        if dump_features:
            feat, _ = model._sample_path(states, bev["f"])
            out["bev_feat"].append(feat[mask].half().cpu().numpy())
    if hook is not None:
        hook.remove()
    return {k: np.concatenate(v) for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-stride", type=int, default=4, help="subsample train windows")
    ap.add_argument("--dump-features", action="store_true",
                    help="also dump the per-cell penultimate BEV feature (fp16), for head refits")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    configs, model = load_from_checkpoint(args.checkpoint, device)

    out = {}
    splits = [("train", configs.DATASET.TRAIN_MISSIONS, args.train_stride),
              ("valid", configs.DATASET.VALID_MISSIONS, 1)]
    for split, missions, stride in splits:
        for mission in tqdm(missions, desc=split):
            res = run_mission(configs, model, mission, device, args.batch_size, args.workers, stride, args.dump_features)
            if res is None:
                continue
            n = len(res["pred_mu"])
            res["mission"] = np.full(n, mission)
            res["split"] = np.full(n, split)
            for k, v in res.items():
                out.setdefault(k, []).append(v)

    merged = {k: np.concatenate(v) for k, v in out.items()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **merged)
    n_tr, n_va = (merged["split"] == "train").sum(), (merged["split"] == "valid").sum()
    print(f"{n_tr} train / {n_va} valid cells -> {args.out}")


if __name__ == "__main__":
    main()
