"""Sidecar to features/extract.py: per-pixel encoder context features (no depth) of the last input frame.

One `<mission>_imgfeat.npz` per cached mission, joined in eval/retiered.load_all.
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from traversability_network.adaptations import mission_ids
from traversability_network.adaptations.grandtour_dataset import GrandTourDataset

from ood_detection.common.bev_cells import sample_cells
from ood_detection.common.checkpoint import image_context, load_checkpoint
from ood_detection.common.windows import anchor_timestamp

SUFFIX = "_imgfeat.npz"


def cached_missions(features_dir):
    """Every mission the window cache holds."""
    paths = sorted(p for p in glob.glob(os.path.join(features_dir, "*.npz"))
                   if not p.endswith(("_cellmaps.npz", SUFFIX)))
    missions = {str(m) for p in paths for m in np.load(p, allow_pickle=True)["mission"]}
    return sorted(missions)


@torch.no_grad()
def extract_mission(configs, model, mission, device, batch_size, workers, n_keep):
    # whole mission regardless of tier spans, to match the cached window layout
    ds = GrandTourDataset(configs, [mission], train=False, time_ranges={mission: [[None, None]]})
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    cells, timestamps = [], []
    for batch in loader:
        color, *rest = (x.to(device) for x in batch)
        image_feats, _, _ = mission_ids.split(rest[9:], *mission_ids.flags(configs))
        B, T = color.shape[:2]
        feats = image_context(model, color, image_feats)
        feats = feats.view(B, T, *feats.shape[1:])[:, -1]
        ones = torch.ones(B, *feats.shape[2:], dtype=torch.bool, device=device)
        keep, _ = sample_cells(feats, ones, n_keep)
        cells.append(keep.half().cpu().numpy())
        timestamps += [anchor_timestamp(ds, len(timestamps) + b) for b in range(B)]

    return {"mission": np.array([mission] * len(timestamps)),
            "timestamp": np.array(timestamps),
            "img_cells": np.concatenate(cells)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features-dir", required=True, help="output dir of features.extract")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--missions", nargs="+", default=None, help="default: all cached ones")
    ap.add_argument("--n-keep", type=int, default=512, help="pixels cached per window")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    configs, model = load_checkpoint(args.cfg, args.checkpoint, device)

    missions = args.missions or cached_missions(args.features_dir)
    for mission in tqdm(missions):
        path = os.path.join(args.features_dir, f"{mission}{SUFFIX}")
        if os.path.exists(path) and not args.overwrite:
            continue
        out = extract_mission(configs, model, mission, device, args.batch_size, args.workers,
                              args.n_keep)
        if out is None:
            print(f"{mission}: no windows, skipping")
            continue
        np.savez(path, **out)
        print(f"{mission}: {len(out['timestamp'])} windows -> {path}")


if __name__ == "__main__":
    main()
