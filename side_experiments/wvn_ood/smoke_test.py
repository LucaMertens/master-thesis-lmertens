"""One-batch smoke test of the WVN gate.

Run: uv run python side_experiments/wvn_ood/smoke_test.py"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_wvn  # registers the WVN config section
from traversability_network.adaptations.grandtour_dataset import GrandTourDataset
from wvn_module import WvnTrainingModule


def save_gray(path, arr, normalize=True):
    a = arr.detach().float().cpu().numpy()
    if normalize:
        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    cv2.imwrite(path, (np.clip(a, 0, 1) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_file", default="side_experiments/wvn_ood/wvn_dino.yaml")
    ap.add_argument("--mission", default="ALB-2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batches", type=int, default=3)
    args = ap.parse_args()

    configs = train_wvn.tn_config.get_cfg(args.cfg_file)
    configs.TRAINING.COMPILE = False
    torch.manual_seed(0)

    dataset = GrandTourDataset(configs, [args.mission], train=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    module = WvnTrainingModule(configs).to(args.device).eval()

    out_dir = "side_experiments/wvn_ood/output/smoke"
    os.makedirs(out_dir, exist_ok=True)
    it = iter(loader)
    for k in range(args.batches):
        batch = [t.to(args.device) for t in next(it)]
        (color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics, path,
         target_trav, trav_weights, depth_target, depth_mask, image_feats) = batch
        with torch.no_grad():
            trav_map, mu_logits, pred_depth, debug = module.model(
                color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics,
                depth_target, image_feats=image_feats)
            print(f"batch {k}: z_max sweep (labeled vs unlabeled visible medians, gate frac)")
            for z_max in [1.5, 0.5, 0.1]:
                module.wvn_z_max = z_max
                gate, score_bev, den = module._wvn_gate(
                    image_feats, inv_intrinsics, extrinsics, depth_target, depth_mask, path)
                labeled = module._path_cell_mask(path)
                vis = den > module.wvn_vis_min
                near = module._near_mask(path)
                lab_s = score_bev[labeled & vis & near]
                unl_s = score_bev[vis & ~labeled & near]
                print(f"  z_max {z_max}: vis {vis.float().mean():.3f}  "
                      f"labeled med {lab_s.median():.0f} q95 {lab_s.quantile(0.95):.0f}  "
                      f"unlabeled med {unl_s.median():.0f}  gate frac {gate.float().mean():.4f} "
                      f"({int(gate.sum())} cells)")
            module.wvn_z_max = configs.WVN.Z_MAX
            gate, score_bev, den = module._wvn_gate(
                image_feats, inv_intrinsics, extrinsics, depth_target, depth_mask, path)
            labeled = module._path_cell_mask(path)
        wvn_loss = (trav_map[:, 0].square() * gate).sum() / gate.sum().clamp(min=1)
        print(f"  wvn_loss {wvn_loss:.4f} on {int(gate.sum())} cells (untrained net)")

        for b in range(min(2, gate.shape[0])):
            row = torch.cat([
                score_bev[b] / (score_bev[b].max() + 1e-6),
                den[b] / (den[b].max() + 1e-6),
                gate[b].float(),
                labeled[b].float(),
            ], dim=1)
            save_gray(f"{out_dir}/b{k}_{b}_score_den_gate_labeled.png", row, normalize=False)

        # camera view of the anchor frame for context
        img = (color_img[0, -1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(f"{out_dir}/b{k}_anchor.jpg", img[..., ::-1])
    print(f"panels -> {out_dir}")


if __name__ == "__main__":
    main()
