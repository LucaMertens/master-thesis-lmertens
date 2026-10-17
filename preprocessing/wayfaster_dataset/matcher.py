"""LoFTR dense matching and depth-based relative pose for the WayFASTER frames."""

import cv2
import numpy as np
import torch
from kornia.feature import LoFTR

UPSCALE = 2  # 320x180 -> 640x360, both divisible by 8 as LoFTR requires


class Matcher:
    def __init__(self, device="cuda", conf=0.5):
        self.model = LoFTR(pretrained="outdoor").eval().to(device)
        self.device = device
        self.conf = conf

    def _prep(self, path):
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(g).float()[None, None].to(self.device) / 255.0

    @torch.no_grad()
    def match(self, path_a, path_b):
        out = self.model({"image0": self._prep(path_a), "image1": self._prep(path_b)})
        c = out["confidence"].cpu().numpy()
        keep = c >= self.conf
        return (
            out["keypoints0"].cpu().numpy()[keep] / UPSCALE,
            out["keypoints1"].cpu().numpy()[keep] / UPSCALE,
            c[keep],
        )


def relative_pose(matcher, img_a, depth_a, img_b, K, min_inliers=20, reproj=3.0):
    """T_camB_from_camA via LoFTR matches plus A's depth (PnP RANSAC)."""
    pa, pb, _ = matcher.match(img_a, img_b)
    if len(pa) < min_inliers:
        return None

    depth = cv2.imread(depth_a, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    u = np.clip(np.round(pa[:, 0]).astype(int), 0, depth.shape[1] - 1)
    v = np.clip(np.round(pa[:, 1]).astype(int), 0, depth.shape[0] - 1)
    z = depth[v, u]
    ok = (z > 0.3) & (z < 20.0)
    if ok.sum() < min_inliers:
        return None

    obj = np.stack(
        [(pa[ok, 0] - K[0, 2]) * z[ok] / K[0, 0], (pa[ok, 1] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1
    ).astype(np.float32)
    ok2, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, pb[ok].astype(np.float32), K, None,
        reprojectionError=reproj, iterationsCount=5000, confidence=0.9999,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok2 or inl is None or len(inl) < min_inliers:
        return None

    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, len(inl), int(ok.sum())
