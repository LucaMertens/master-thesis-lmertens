"""Pick a diverse set of eval frames per mission for the labeling tool.

Run: uv run python side_experiments/wvn_ood/eval_labeling/select_frames.py"""

import argparse
import json
import os

import cv2
import numpy as np
import yaml

SIZE = (480, 360)


def pooled_features(cache_dir):
    """(ts_ns, L2-normalized mean-pooled features (N, D)) in cache row order."""
    meta = np.load(os.path.join(cache_dir, "meta.npz"), allow_pickle=True)
    feats = np.load(os.path.join(cache_dir, "features.npy"), mmap_mode="r")
    pooled = np.empty((feats.shape[0], feats.shape[1]), np.float32)
    for lo in range(0, feats.shape[0], 256):
        pooled[lo : lo + 256] = feats[lo : lo + 256].astype(np.float32).mean(axis=(2, 3))
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-8
    return meta["ts_ns"].astype(np.int64), pooled


def k_center(X, k, seed_from_mean=True):
    """Greedy k-center (cosine distance) row indices."""
    center = X.mean(0)
    center /= np.linalg.norm(center) + 1e-8
    d = 1 - X @ center
    picks = []
    for _ in range(min(k, len(X))):
        i = int(np.argmax(d))
        picks.append(i)
        d = np.minimum(d, 1 - X @ X[i])
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="side_experiments/wvn_ood/wvn_dino.yaml")
    ap.add_argument("--out", default="side_experiments/wvn_ood/output/eval_labeling")
    ap.add_argument("--per-mission", type=int, default=15)
    ap.add_argument("--far-ood", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    ds = cfg["DATASET"]
    missions = ds["VALID_MISSIONS"] + ds["NEAR_OOD_MISSIONS"]
    if args.far_ood:
        missions += ds["FAR_OOD_MISSIONS"]

    frames_dir = os.path.join(args.out, "frames")
    labels_dir = os.path.join(args.out, "labels")
    os.makedirs(frames_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "frames.json")
    old = {e["name"]: e for e in json.load(open(manifest_path))} if os.path.exists(manifest_path) else {}
    entries = {n: e for n, e in old.items()
               if os.path.exists(os.path.join(labels_dir, n + ".png"))}  # keep labeled

    for mission in missions:
        ts_ns, X = pooled_features(os.path.join(ds["DINO_CACHE_ROOT"], mission))
        lo, hi = int(0.1 * len(ts_ns)), int(0.9 * len(ts_ns))
        picks = k_center(X[lo:hi], args.per_mission)
        kept = sum(1 for e in entries.values() if e["mission"] == mission)

        fc = np.load(os.path.join(ds["DATA_ROOT"], mission, "image_manifest.npz"))["front_center"]
        order = np.argsort(fc["ts_ns"].astype(np.int64))
        m_ts = fc["ts_ns"].astype(np.int64)[order]
        root = os.path.join(ds["RESIZED_ROOT"], mission)
        if not os.path.isdir(root):
            root = os.path.join(ds["DATA_ROOT"], mission)

        added = 0
        for p in picks:
            if kept + added >= args.per_mission:
                break
            t = int(ts_ns[lo + p])
            name = f"{mission}_{t}"
            if name in entries:
                continue
            j = int(np.searchsorted(m_ts, t))
            assert m_ts[j] == t, f"{name}: cache ts not in manifest"
            img = cv2.imread(os.path.join(root, fc["path"][order[j]]), -1)
            assert img is not None
            if (img.shape[1], img.shape[0]) != SIZE:
                img = cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(frames_dir, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            entries[name] = {"name": name, "mission": mission, "ts_ns": t, "w": SIZE[0], "h": SIZE[1]}
            added += 1
        print(f"{mission}: kept {kept} labeled, added {added} k-center picks")

    ordered = sorted(entries.values(), key=lambda e: (e["mission"], e["ts_ns"]))
    json.dump(ordered, open(manifest_path, "w"), indent=1)
    print(f"{len(ordered)} frames in {manifest_path}")


if __name__ == "__main__":
    main()
