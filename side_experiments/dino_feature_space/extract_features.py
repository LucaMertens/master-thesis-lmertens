"""Frozen DINOv3 (HF Hub) feature extraction over the whole GrandTour dataset,
subsampled per mission, tagged with mission/tier for feature-space visualization
(see plots.ipynb): are ID/near-OOD/far-OOD missions separable in DINO space?"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import AutoModel

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# mirrors ood_detection/common/splits.py TIER_MISSIONS
TIER_MISSIONS = {
    "near_forest": ["KAEB-1", "ROOT-1"],
    "near_gravel": ["EIG-1", "EIG-2", "PIL-2"],
    "far_snow": ["SNOW-1", "SNOW-2", "SNOW-3"],
    "far_arc": ["ARC-1", "ARC-2", "ARC-3", "ARC-4"],
}


def mission_tiers(cfg):
    ds = cfg["DATASET"]
    near = set(ds["NEAR_OOD_MISSIONS"])
    far = set(ds["FAR_OOD_MISSIONS"])
    expected_near = set(TIER_MISSIONS["near_forest"]) | set(TIER_MISSIONS["near_gravel"])
    expected_far = set(TIER_MISSIONS["far_snow"]) | set(TIER_MISSIONS["far_arc"])
    assert near == expected_near, f"config NEAR_OOD_MISSIONS {near} != expected {expected_near}"
    assert far == expected_far, f"config FAR_OOD_MISSIONS {far} != expected {expected_far}"
    tiers = {"train": list(ds["TRAIN_MISSIONS"]), "id_heldout": list(ds["VALID_MISSIONS"])}
    tiers.update(TIER_MISSIONS)
    return tiers


def sample_paths(ds, mission, n_frames, rng):
    """Evenly-spaced (deterministic given n_frames) subsample of a mission's
    front_center frames, resolved against RESIZED_ROOT."""
    manifest = np.load(os.path.join(ds["DATA_ROOT"], mission, "image_manifest.npz"))
    fc = manifest["front_center"]
    order = np.argsort(fc["ts_ns"])
    fc = fc[order]
    idx = np.linspace(0, len(fc) - 1, min(n_frames, len(fc))).round().astype(int)
    idx = np.unique(idx)
    resized_root = os.path.join(ds["RESIZED_ROOT"], mission)
    root = resized_root if os.path.isdir(resized_root) else os.path.join(ds["DATA_ROOT"], mission)
    paths = [os.path.join(root, p) for p in fc["path"][idx]]
    ts = fc["ts_ns"][idx].astype(np.float64) / 1e9
    return paths, ts


def load_dino(name, device):
    model = AutoModel.from_pretrained(name).eval().to(device)
    return model


@torch.no_grad()
def dino_pooled_features(model, images, patch, num_prefix_tokens, device):
    """images: (B, 3, H, W) in [0, 1] RGB. Mean-pools patch tokens (drops CLS +
    register tokens) of the last hidden state -> (B, D)."""
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    H, W = images.shape[-2:]
    Hc, Wc = H // patch * patch, W // patch * patch
    x = images if (Hc, Wc) == (H, W) else F.interpolate(images, size=(Hc, Wc), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
        out = model(pixel_values=x).last_hidden_state.float()
    return out[:, num_prefix_tokens:].mean(dim=1)


def load_batch(paths, size, device):
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        assert img is not None, p
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if img.shape[:2] != size:
            img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(device)
    return x


def extract_mission(ds, model, patch, num_prefix, mission, n_frames, rng, device, batch_size, input_size):
    paths, ts = sample_paths(ds, mission, n_frames, rng)
    feats = []
    for k in range(0, len(paths), batch_size):
        x = load_batch(paths[k : k + batch_size], input_size, device)
        feats.append(dino_pooled_features(model, x, patch, num_prefix, device).cpu().numpy())
    return np.concatenate(feats), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="training/traversability_network/configs/grandtour.yaml")
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--frames-per-mission", type=int, default=200)
    ap.add_argument("--out", default="side_experiments/dino_feature_space/output/features.npz")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    ds = cfg["DATASET"]
    input_size = (cfg["MODEL"]["INPUT_SIZE"][1], cfg["MODEL"]["INPUT_SIZE"][0])  # (H, W)
    tiers = mission_tiers(cfg)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)
    model = load_dino(args.model, device)
    patch = model.config.patch_size
    num_prefix = 1 + model.config.num_register_tokens

    feats, missions, tier_labels, timestamps = [], [], [], []
    for tier, missions_in_tier in tiers.items():
        for mission in tqdm(missions_in_tier, desc=tier):
            f, ts = extract_mission(
                ds, model, patch, num_prefix, mission, args.frames_per_mission, rng, device, args.batch_size, input_size
            )
            feats.append(f)
            missions += [mission] * len(f)
            tier_labels += [tier] * len(f)
            timestamps.append(ts)
            print(f"  {mission}: {len(f)} frames")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        features=np.concatenate(feats),
        mission=np.array(missions),
        tier=np.array(tier_labels),
        timestamp=np.concatenate(timestamps),
        model=args.model,
    )
    print(f"{len(missions)} frames -> {args.out}")


if __name__ == "__main__":
    main()
