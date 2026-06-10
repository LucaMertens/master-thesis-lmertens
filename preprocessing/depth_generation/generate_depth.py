#!/usr/bin/env python
"""Run FoundationStereo on rectified GrandTour stereo pairs.

Writes 16-bit mm depth PNGs and depth_manifest.npz per mission.
Run via run.sh (Docker); inputs from grandtour_extraction.pipeline.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "FoundationStereo")
)
from core.foundation_stereo import FoundationStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402


def load_model(ckpt_path: str) -> FoundationStereo:
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "cfg.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"{cfg_path} not found — download the 23-51-11 folder (see README)."
        )
    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = 32
    cfg["mixed_precision"] = True
    model = FoundationStereo(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    return model


def run_stereo(
    model: FoundationStereo,
    left_img: np.ndarray,
    right_img: np.ndarray,
    scale: float = 0.5,
) -> np.ndarray:
    """Disparity map (float32, pixels) at original resolution."""
    if left_img.ndim == 2:
        left_img = np.stack([left_img] * 3, axis=-1)
        right_img = np.stack([right_img] * 3, axis=-1)

    h_orig, w_orig = left_img.shape[:2]
    img0 = cv2.resize(left_img, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(right_img, fx=scale, fy=scale, dsize=None)
    h, w = img0.shape[:2]

    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.amp.autocast(device_type="cuda", enabled=True):
        disp = model.forward(img0_t, img1_t, iters=32, test_mode=True)
    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(h, w)

    # disparity scales linearly with width
    disp_full = (
        cv2.resize(disp, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR) / scale
    )
    return disp_full.astype(np.float32)


def disparity_to_depth(disparity: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 1e-3
    depth[valid] = fx * baseline / disparity[valid]
    return depth


def match_pairs(left_ts, right_ts, tol=0.01):
    idx = np.searchsorted(right_ts, left_ts)
    idx = np.clip(idx, 1, len(right_ts) - 1)
    choose_left = (left_ts - right_ts[idx - 1]) <= (right_ts[idx] - left_ts)
    nearest = np.where(choose_left, idx - 1, idx)
    return nearest, np.abs(right_ts[nearest] - left_ts) <= tol


def _require(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — run `uv run python -m grandtour_extraction.pipeline --no-labels` first."
        )
    return path


def process_mission(
    model: FoundationStereo, mission_dir: str, scale: float = 0.5, stride: int = 1
) -> None:
    manifest = np.load(_require(os.path.join(mission_dir, "image_manifest.npz")))
    ref_arr, partner_arr = manifest["front_center"], manifest["front_right"]
    k_ref = np.load(_require(os.path.join(mission_dir, "intrinsics.npy")))
    stereo = np.load(_require(os.path.join(mission_dir, "stereo_extrinsics.npz")))
    baseline = float(stereo["baseline"])
    fx = float(k_ref[0, 0])

    ref_ts = ref_arr["ts_ns"].astype(np.float64) / 1e9
    partner_ts = partner_arr["ts_ns"].astype(np.float64) / 1e9
    ref_ord = np.argsort(ref_ts)
    ref_ts = ref_ts[ref_ord]
    ref_arr = ref_arr[ref_ord]
    p_ord = np.argsort(partner_ts)
    partner_ts = partner_ts[p_ord]
    partner_arr = partner_arr[p_ord]
    ref_ts = ref_ts[::stride]
    ref_arr = ref_arr[::stride]
    nearest, ok = match_pairs(ref_ts, partner_ts)

    depth_dir = os.path.join(mission_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    rows = []

    for i in tqdm(range(len(ref_ts)), desc=os.path.basename(mission_dir.rstrip("/"))):
        if not ok[i]:
            continue
        # Grayscale on both sides: the reference is colour, the partner is mono.
        ref_img = cv2.imread(
            os.path.join(mission_dir, ref_arr["path"][i]), cv2.IMREAD_GRAYSCALE
        )
        partner_img = cv2.imread(
            os.path.join(mission_dir, partner_arr["path"][nearest[i]]),
            cv2.IMREAD_GRAYSCALE,
        )
        disp = run_stereo(model, ref_img, partner_img, scale=scale)
        depth = disparity_to_depth(disp, fx, baseline)
        # zero the unreliable top border: both the rectification black region
        # and the InputPadder top edge emit spurious near-depth. Crop the top
        # band and any black pixels (+ halo) elsewhere.
        depth[:16] = 0.0
        border = cv2.dilate((ref_img < 10).astype(np.uint8), np.ones((17, 17), np.uint8))
        depth[border.astype(bool)] = 0.0
        ts_ns = int(ref_arr["ts_ns"][i])
        rel = f"depth/{ts_ns}.png"
        cv2.imwrite(
            os.path.join(mission_dir, rel),
            (depth * 1000.0).clip(0, 65535).astype(np.uint16),
        )
        rows.append((ts_ns, rel))

    np.savez(
        os.path.join(mission_dir, "depth_manifest.npz"),
        depth=np.array(rows, dtype=[("ts_ns", "i8"), ("path", "U128")]),
    )
    print(f"Wrote {len(rows)} depth maps -> {depth_dir}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", default="pretrained_models/23-51-11/model_best_bp2.pth")
    p.add_argument("--missions", nargs="+", default=["/data/ETH-1"])
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every n-th left image (default: 1, i.e. every image)",
    )
    args = p.parse_args()
    assert args.scale <= 1, "--scale must be <= 1"
    assert args.stride >= 1, "--stride must be >= 1"
    torch.autograd.set_grad_enabled(False)
    model = load_model(args.ckpt)
    for m in args.missions:
        process_mission(model, m, scale=args.scale, stride=args.stride)


if __name__ == "__main__":
    main()
