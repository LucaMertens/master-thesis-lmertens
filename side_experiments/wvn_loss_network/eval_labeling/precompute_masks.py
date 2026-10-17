"""Precompute segment proposals for the labeling tool's click-to-fill brush."""

import argparse
import json
import os

import cv2
import numpy as np

GRID = 32        # px, split large uncovered components
BIG_COMP = 5000  # px, components above this get grid-split
FALLBACK_GRID = 32


def build_id_map(masks, h, w):
    idmap = np.zeros((h, w), np.uint32)
    order = np.argsort([-int(m.sum()) for m in masks])
    next_id = 1
    for k in order:
        idmap[masks[k]] = next_id
        next_id += 1
    # uncovered pixels: components, big ones grid-split
    n, comp = cv2.connectedComponents((idmap == 0).astype(np.uint8), connectivity=4)
    ys, xs = np.mgrid[0:h, 0:w]
    cell = (ys // GRID) * (w // GRID + 1) + xs // GRID
    for c in range(1, n):
        sel = comp == c
        if sel.sum() > BIG_COMP:
            sub = cell[sel]
            for u in np.unique(sub):
                m = sel & (cell == u)
                idmap[m] = next_id
                next_id += 1
        else:
            idmap[sel] = next_id
            next_id += 1
    return idmap


def grid_id_map(h, w, step=FALLBACK_GRID):
    ys, xs = np.mgrid[0:h, 0:w]
    return ((ys // step) * (w // step + 1) + xs // step + 1).astype(np.uint32)


def encode(idmap):
    rgb = np.stack([idmap & 255, (idmap >> 8) & 255, (idmap >> 16) & 255], -1).astype(np.uint8)
    return rgb[..., ::-1]  # cv2 writes BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="side_experiments/wvn_loss_network/output/eval_labeling")
    ap.add_argument("--model", default="facebook/sam-vit-large")
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--points-per-batch", type=int, default=64)
    # relaxed SAM filters, forest floors get few masks otherwise
    ap.add_argument("--pred-iou-thresh", type=float, default=0.72)
    ap.add_argument("--stability-thresh", type=float, default=0.88)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    frames = json.load(open(os.path.join(args.out, "frames.json")))
    seg_dir = os.path.join(args.out, "segments")
    os.makedirs(seg_dir, exist_ok=True)

    generator = None
    try:
        from transformers import pipeline

        generator = pipeline("mask-generation", model=args.model, device=args.device)
        print(f"SAM backend: {args.model} on {args.device}")
    except Exception as e:
        print(f"SAM unavailable ({e}); falling back to {FALLBACK_GRID}px grid segments")

    for i, e in enumerate(frames):
        dst = os.path.join(seg_dir, e["name"] + ".png")
        if os.path.exists(dst) and not args.force:
            continue
        img = cv2.imread(os.path.join(args.out, "frames", e["name"] + ".jpg"))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if generator is not None:
            from PIL import Image

            out = generator(
                Image.fromarray(rgb), points_per_batch=args.points_per_batch,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_thresh,
            )
            masks = [np.asarray(m, bool) for m in out["masks"]]
            idmap = build_id_map(masks, *rgb.shape[:2])
            n_seg = int(idmap.max())
        else:
            idmap = grid_id_map(*rgb.shape[:2])
            n_seg = int(idmap.max())
        cv2.imwrite(dst, encode(idmap))
        print(f"[{i + 1}/{len(frames)}] {e['name']}: {n_seg} segments", flush=True)
    print("done")


if __name__ == "__main__":
    main()
