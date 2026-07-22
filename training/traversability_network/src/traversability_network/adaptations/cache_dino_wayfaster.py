"""Caches frozen DINO patch features for the original WayFASTER dataset (CSV bag layout) as fp16 memmaps in images.csv row order."""

import argparse
import os
import pathlib

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ood_detection.common.dino import dino_patch_features, load_dino, patch_size


def load_frame(path, input_size):
    img = cv2.imread(path, -1)
    assert img is not None, f"unreadable image: {path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (img.shape[1], img.shape[0]) != tuple(input_size):
        img = cv2.resize(img, tuple(input_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def cache_bag(bag_dir, out_dir, model, patch, input_size, device, batch):
    paths = [str(bag_dir / p) for p in pd.read_csv(bag_dir / "images.csv")["image"]]
    h, w = input_size[1] // patch, input_size[0] // patch
    probe = dino_patch_features(model, load_frame(paths[0], input_size)[None].to(device), patch)
    feat_dim = probe.shape[1]
    assert probe.shape[2:] == (h, w), f"unexpected patch grid {tuple(probe.shape)}"

    out_dir.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(out_dir / "features.npy", mode="w+", dtype=np.float16,
                                      shape=(len(paths), feat_dim, h, w))
    feats_flip = np.lib.format.open_memmap(out_dir / "features_flip.npy", mode="w+", dtype=np.float16,
                                           shape=(len(paths), feat_dim, h, w))
    for lo in range(0, len(paths), batch):
        chunk = paths[lo : lo + batch]
        images = torch.stack([load_frame(p, input_size) for p in chunk]).to(device)
        feats[lo : lo + len(chunk)] = dino_patch_features(model, images, patch).half().cpu().numpy()
        feats_flip[lo : lo + len(chunk)] = dino_patch_features(model, images.flip(-1), patch).half().cpu().numpy()
    feats.flush()
    feats_flip.flush()
    return len(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+",
                    default=["/datahome/lmertens/wayfaster_dataset/realsense/data_train",
                             "/datahome/lmertens/wayfaster_dataset/realsense/data_valid"])
    ap.add_argument("--cache-root", default="/datalocal/lmertens/wayfaster_dino")
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--input-size", type=int, nargs=2, default=[320, 176], help="(W, H), patch multiple")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model = load_dino(args.model, device=args.device)
    patch = patch_size(model)
    cache_root = pathlib.Path(args.cache_root)

    for data_dir in args.data_dirs:
        data_dir = pathlib.Path(data_dir)
        bags = pd.read_csv(data_dir / "rosbags.csv").iloc[:, 0]
        for bag in tqdm(list(bags), desc=os.path.basename(data_dir)):
            out_dir = cache_root / bag
            if (out_dir / "features_flip.npy").exists():
                continue
            n = cache_bag(data_dir / bag, out_dir, model, patch, tuple(args.input_size), args.device, args.batch)
            tqdm.write(f"{bag}: {n} frames -> {out_dir}")


if __name__ == "__main__":
    main()
