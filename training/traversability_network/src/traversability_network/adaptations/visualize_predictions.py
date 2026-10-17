"""Qualitative checkpoint check: predicted mu/nu and labelled path overlaid on the camera image."""
import argparse
import os

import cv2
import numpy as np
import torch

from ..config import get_cfg
from ..network.trainer import TrainingModule
from . import mission_ids
from .grandtour_dataset import GrandTourDataset
from .overlay import overlay_bev_on_image, overlay_path_on_image


def tensor_to_bgr(color_img):
    """(3, H, W) float [0,1] RGB -> (H, W, 3) uint8 BGR."""
    rgb = (color_img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def label_panel(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def load_model(cfg_file, checkpoint, device):
    configs = get_cfg(cfg_file)
    model = TrainingModule(configs)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"non-strict load: missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    return configs, model


@torch.no_grad()
def render_sample(model, batch, device, grid_bounds):
    color_img, depth_voxel, intrinsics_full, intrinsics, extrinsics, states, target_trav, trav_weights, depth_target, depth_mask, *extra = batch
    b = tuple(x.unsqueeze(0).to(device) for x in (color_img, depth_voxel, intrinsics_full, intrinsics, extrinsics, depth_target))
    image_feats, _, _ = mission_ids.split(extra, *model.extra_flags)
    if image_feats is not None:
        image_feats = image_feats.unsqueeze(0).to(device)
    trav_map, _, _, _ = model.model(*b, image_feats=image_feats)
    trav_map = trav_map[0].cpu().numpy()  # (2, Hd, Wd)

    K, ext = intrinsics[-1].numpy(), extrinsics[-1].numpy()
    rgb = tensor_to_bgr(color_img[-1])

    ground_z = TrainingModule._ground_plane(states)
    mu_panel = label_panel(overlay_bev_on_image(rgb, trav_map[0], K, ext, grid_bounds, ground_z=ground_z), "predicted mu")
    nu_panel = label_panel(overlay_bev_on_image(rgb, trav_map[1], K, ext, grid_bounds, ground_z=ground_z), "predicted nu")
    gt_panel = label_panel(overlay_path_on_image(rgb, states, target_trav, trav_weights, K, ext), "labelled mu (path)")
    return np.concatenate([mu_panel, nu_panel, gt_panel], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Overlay predicted/labelled traversability on the camera image.")
    parser.add_argument("--cfg_file", required=True, help="config the checkpoint was trained with")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default="vis_predictions")
    parser.add_argument("--n_samples", type=int, default=8, help="images saved per split")
    parser.add_argument("--n_lowmu", type=int, default=4, help="of n_samples, reserved for windows with a labelled low-mu path point")
    parser.add_argument("--lowmu_thresh", type=float, default=0.5)
    parser.add_argument("--max_scan", type=int, default=200, help="cap on windows inspected per split")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], choices=["train", "valid"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configs, model = load_model(args.cfg_file, args.checkpoint, args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for split in args.splits:
        missions = configs.DATASET.TRAIN_MISSIONS if split == "train" else configs.DATASET.VALID_MISSIONS
        dataset = GrandTourDataset(configs, missions, train=False)
        order = rng.permutation(len(dataset))[: args.max_scan]

        n_saved, n_lowmu_saved = 0, 0
        for idx in order:
            if n_saved >= args.n_samples:
                break
            batch = dataset[int(idx)]
            target_trav, trav_weights = batch[6], batch[7]
            has_lowmu = bool(((target_trav[:, 0] < args.lowmu_thresh) & (trav_weights[:, 0] > 0)).any())
            n_other_slots = args.n_samples - args.n_lowmu
            if has_lowmu and n_lowmu_saved >= args.n_lowmu:
                continue
            if not has_lowmu and (n_saved - n_lowmu_saved) >= n_other_slots:
                continue

            panel = render_sample(model, batch, args.device, configs.MODEL.GRID_BOUNDS)
            tag = "lowmu" if has_lowmu else "random"
            fname = os.path.join(args.out_dir, f"{split}_{n_saved:03d}_{tag}_idx{idx}.png")
            cv2.imwrite(fname, panel)

            n_saved += 1
            n_lowmu_saved += int(has_lowmu)

        print(f"{split}: saved {n_saved} images ({n_lowmu_saved} with a labelled low-mu path point) -> {args.out_dir}")


if __name__ == "__main__":
    main()
