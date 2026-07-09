"""Step 1 of the OOD pipeline: run a trained checkpoint over every mission tier
(id_train, id_heldout, near_forest, near_gravel, far_snow, far_arc) and cache, per
window: the WayFASTER penultimate BEV feature, a DINO frame feature, the RGB+D and
RGB-only mu predictions (for the modality-disagreement score), the evidential
uncertainty (if the checkpoint has an evidential head), and the true-mu error over
visible+observable path cells (for the RQ3 failure-mode handoff).

One .npz per tier written to --out-dir; ood_detection.eval.run reads them back.
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from traversability_network.data.grandtour_dataset import GrandTourDataset

from ood_detection.common.checkpoint import BEVFeatureHook, load_checkpoint
from ood_detection.common.dino import dino_pooled_features, load_dino, patch_size
from ood_detection.common.splits import all_tiers
from ood_detection.common.windows import anchor_timestamp

FIELDS = (
    "timestamp", "bev_feat", "bev_feat_rgbonly", "dino_feat",
    "mu_pred", "mu_pred_rgbonly", "mu_true_min", "mu_true_mean", "mu_err_mean",
    "n_visible", "evidential_unc", "evidential_unc_rgbonly",
)


@torch.no_grad()
def extract_mission(configs, model, dino, dino_patch, mission, device, batch_size, workers):
    ds = GrandTourDataset(configs, [mission], train=False)
    if len(ds) == 0:
        return None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    hook = BEVFeatureHook(model)
    rows = {k: [] for k in FIELDS}

    idx0 = 0
    for batch in loader:
        color, depth_voxel, K_full, K, ext, states, target_trav, trav_weights, depth_target, _ = (
            x.to(device) for x in batch
        )
        trav_map, aux, _, _ = model.model(color, depth_voxel, K_full, K, ext, depth_target)
        bev = hook.features.mean(dim=(2, 3))

        trav_map_rgbonly, aux_rgbonly, _, _ = model.model(color, torch.zeros_like(depth_voxel), K_full, K, ext, depth_target)
        bev_rgbonly = hook.features.mean(dim=(2, 3))

        pred, mask = model._sample_path(states, trav_map)
        pred_rgbonly, _ = model._sample_path(states, trav_map_rgbonly)
        vis = mask & (trav_weights[..., 0] > 0)
        dino_feat = dino_pooled_features(dino, color[:, -1], dino_patch)

        unc = unc_rgbonly = None
        if configs.MODEL.EVIDENTIAL:
            aux_path, _ = model._sample_path(states, aux)
            aux_path_rgbonly, _ = model._sample_path(states, aux_rgbonly)
            unc = aux_path[..., 2] / (aux_path[..., 0] * (aux_path[..., 1] - 1).clamp_min(1e-3))
            unc_rgbonly = aux_path_rgbonly[..., 2] / (aux_path_rgbonly[..., 0] * (aux_path_rgbonly[..., 1] - 1).clamp_min(1e-3))

        B = color.shape[0]
        for b in range(B):
            m = vis[b]
            n = int(m.sum())
            rows["n_visible"].append(n)
            rows["timestamp"].append(anchor_timestamp(ds, idx0 + b))
            rows["bev_feat"].append(bev[b].cpu().numpy())
            rows["bev_feat_rgbonly"].append(bev_rgbonly[b].cpu().numpy())
            rows["dino_feat"].append(dino_feat[b].cpu().numpy())
            if n == 0:
                for k in ("mu_pred", "mu_pred_rgbonly", "mu_true_min", "mu_true_mean",
                          "mu_err_mean", "evidential_unc", "evidential_unc_rgbonly"):
                    rows[k].append(float("nan"))
                continue
            p, t = pred[b, m, 0], target_trav[b, m, 0]
            rows["mu_pred"].append(float(p.mean()))
            rows["mu_pred_rgbonly"].append(float(pred_rgbonly[b, m, 0].mean()))
            rows["mu_true_min"].append(float(t.min()))
            rows["mu_true_mean"].append(float(t.mean()))
            rows["mu_err_mean"].append(float((p - t).abs().mean()))
            rows["evidential_unc"].append(float(unc[b, m].mean()) if unc is not None else float("nan"))
            rows["evidential_unc_rgbonly"].append(float(unc_rgbonly[b, m].mean()) if unc_rgbonly is not None else float("nan"))
        idx0 += B

    hook.remove()
    out = {k: np.array(v) for k, v in rows.items() if k not in ("bev_feat", "bev_feat_rgbonly", "dino_feat")}
    out["bev_feat"] = np.stack(rows["bev_feat"])
    out["bev_feat_rgbonly"] = np.stack(rows["bev_feat_rgbonly"])
    out["dino_feat"] = np.stack(rows["dino_feat"])
    out["mission"] = np.array([mission] * len(out["timestamp"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="config the checkpoint was trained with")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dino-model", default="dinov2_vitb14_reg")
    ap.add_argument("--dino-weights", default=None, help="local .pth for gated dinov3_* models")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tiers", nargs="+", default=None, help="default: all tiers, including id_train")
    args = ap.parse_args()

    device = torch.device(args.device)
    configs, model = load_checkpoint(args.cfg, args.checkpoint, device)
    dino = load_dino(args.dino_model, args.dino_weights, device)
    patch = patch_size(dino)

    tiers = all_tiers(configs)
    if args.tiers:
        tiers = {k: v for k, v in tiers.items() if k in args.tiers}

    os.makedirs(args.out_dir, exist_ok=True)
    for tier, missions in tiers.items():
        chunks = [
            extract_mission(configs, model, dino, patch, m, device, args.batch_size, args.workers)
            for m in tqdm(missions, desc=tier)
        ]
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            print(f"{tier}: no windows, skipping")
            continue
        merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        out_path = os.path.join(args.out_dir, f"{tier}.npz")
        np.savez_compressed(out_path, **merged)
        print(f"{tier}: {len(merged['timestamp'])} windows -> {out_path}")


if __name__ == "__main__":
    main()
