"""Caches frozen DINO patch features per mission as fp16 memmaps.

Also caches features of horizontally flipped frames: DINO isn't flip-equivariant
(flip(feat) vs feat(flip) cosine ~0.81), so flip augmentation reads this cache
instead of flipping the feature map. Machine-local training cache, like
cache_depth.py, not a pipeline.py step. Needs a GPU and, for gated DINOv3
weights, a logged-in HF token.
"""

import argparse
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from config.paths import DATALOCAL_ROOT, EXTRACTED_ROOT, ensure_datalocal_root
from data_labeling_pipeline.pipeline_step import is_step_done, log_step, write_step_done
from ood_detection.common.dino import dino_patch_features, load_dino, patch_size
from tqdm import tqdm

VERSION = 2
STEP = "cache_dino_features"

DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_INPUT_SIZE = (480, 352)  # (W, H), must be a patch-size multiple


def _load_frame(path, input_size):
    img = cv2.imread(path, -1)
    assert img is not None, f"unreadable image: {path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if (img.shape[1], img.shape[0]) != tuple(input_size):
        img = cv2.resize(img, tuple(input_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def _meta_matches(out_dir, model_name, input_size):
    meta_path = out_dir / "meta.npz"
    if not meta_path.exists():
        return False
    meta = np.load(meta_path, allow_pickle=True)
    return str(meta["model"]) == model_name and list(meta["input_size"]) == list(input_size)


def cache_mission(
    mission_dir: pathlib.Path,
    cache_root: pathlib.Path,
    model,
    model_name: str,
    input_size,
    device: str,
    batch: int,
    n_workers: int,
    force: bool,
    resized_root: pathlib.Path | None = None,
) -> None:
    if cache_root == DATALOCAL_ROOT or DATALOCAL_ROOT in cache_root.parents:
        ensure_datalocal_root()

    out_dir = cache_root / mission_dir.name
    if not force and is_step_done(out_dir, STEP, VERSION) and _meta_matches(out_dir, model_name, input_size):
        print(f"Skipping {STEP} (up to date): {out_dir}")
        return

    fc = np.load(mission_dir / "image_manifest.npz")["front_center"]
    ts_ns = fc["ts_ns"].astype(np.int64)
    order = np.argsort(ts_ns)
    ts_ns = ts_ns[order]
    assert np.all(np.diff(ts_ns) > 0), f"{mission_dir.name}: duplicate manifest timestamps"

    img_root = mission_dir
    if resized_root and (resized_root / mission_dir.name).is_dir():
        img_root = resized_root / mission_dir.name
    paths = [str(img_root / fc["path"][i]) for i in order]

    patch = patch_size(model)
    W, H = input_size
    assert W % patch == 0 and H % patch == 0, f"input_size {input_size} not a multiple of patch {patch}"
    h, w = H // patch, W // patch
    feat_dim = model.config.hidden_size if hasattr(model, "config") else model.embed_dim

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_feat = out_dir / f"features.npy.tmp{os.getpid()}"
    tmp_flip = out_dir / f"features_flip.npy.tmp{os.getpid()}"
    with log_step(STEP, out_dir):
        shape = (len(paths), feat_dim, h, w)
        feats = np.lib.format.open_memmap(tmp_feat, mode="w+", dtype=np.float16, shape=shape)
        feats_flip = np.lib.format.open_memmap(tmp_flip, mode="w+", dtype=np.float16, shape=shape)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for lo in tqdm(range(0, len(paths), batch), desc=mission_dir.name):
                chunk = paths[lo : lo + batch]
                images = torch.stack(list(pool.map(lambda p: _load_frame(p, input_size), chunk))).to(device)
                fmap = dino_patch_features(model, images, patch)
                assert fmap.shape[2:] == (h, w), f"unexpected patch grid {tuple(fmap.shape)}"
                feats[lo : lo + len(chunk)] = fmap.half().cpu().numpy()
                fmap = dino_patch_features(model, images.flip(-1), patch)
                feats_flip[lo : lo + len(chunk)] = fmap.half().cpu().numpy()
        feats.flush()
        feats_flip.flush()
        del feats, feats_flip
        os.replace(tmp_feat, out_dir / "features.npy")
        os.replace(tmp_flip, out_dir / "features_flip.npy")

        tmp_meta = out_dir / f"meta.npz.tmp{os.getpid()}"
        with open(tmp_meta, "wb") as f:
            np.savez(
                f, ts_ns=ts_ns, model=model_name, input_size=np.array(input_size),
                patch=patch, feat_dim=feat_dim, version=VERSION,
            )
        os.replace(tmp_meta, out_dir / "meta.npz")
        write_step_done(out_dir, STEP, VERSION, count=len(paths), model=model_name)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--extracted-root", type=pathlib.Path, default=EXTRACTED_ROOT)
    p.add_argument("--resized-root", type=pathlib.Path,
                   default=DATALOCAL_ROOT / "grand_tour_dataset_resized",
                   help="pre-resized image root (resize_images.py); falls back to extracted-root")
    p.add_argument("--cache-root", type=pathlib.Path,
                   default=DATALOCAL_ROOT / "grand_tour_dataset_dino",
                   help="DATASET.DINO_CACHE_ROOT")
    p.add_argument("--missions", nargs="+", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--input-size", type=int, nargs=2, default=list(DEFAULT_INPUT_SIZE),
                   metavar=("W", "H"), help="fed to DINO; must be patch-size multiples")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8, help="image-decoding threads")
    p.add_argument("--force", action="store_true", help="Re-run even if up to date.")
    args = p.parse_args()

    model = load_dino(args.model, device=args.device)
    for mission in args.missions:
        cache_mission(
            args.extracted_root / mission, args.cache_root, model, args.model,
            tuple(args.input_size), args.device, args.batch, args.workers,
            args.force, resized_root=args.resized_root,
        )


if __name__ == "__main__":
    main()
