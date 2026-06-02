"""Fisheye stereo rectification for Alphasense front_center (ref, colour) and front_right (mono).

front_center sits left of front_right, so disparity is positive and depth lands directly in the
front_center frame. Single source of truth for maps and intrinsics used across extraction steps.
"""

import pathlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from data_labeling_pipeline.ros1_transforms import get_static_transform, inv
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

REF_CAM = "front_center"      # colour reference; depth is produced in this frame
PARTNER_CAM = "front_right"   # mono partner, mounted to the right of the reference
CAMERAS = (REF_CAM, PARTNER_CAM)
FRAME = {c: f"alphasense_{c}" for c in CAMERAS}

# Rectification zoom. Identical values must be used everywhere; keep them here.
DEFAULT_BALANCE = 0.0
DEFAULT_FOV_SCALE = 1.0


@dataclass
class CamInfo:
    K: np.ndarray            # (3, 3)
    D: np.ndarray            # (4, 1) equidistant coeffs
    size: tuple[int, int]    # (w, h)


@dataclass
class Maps:
    map1: np.ndarray
    map2: np.ndarray
    K: np.ndarray            # rectified/undistorted pinhole K (3, 3)


@dataclass
class Rectifiers:
    ref: Maps                # front_center stereo-rectified (reference, colour)
    partner: Maps            # front_right stereo-rectified (mono)
    baseline: float          # rectified stereo baseline (m), from P2
    R_ref: np.ndarray        # raw front_center -> rect front_center rotation
    R_partner: np.ndarray    # raw front_right  -> rect front_right rotation
    t_base_ref: np.ndarray   # (4, 4) base <- raw front_center
    t_base_partner: np.ndarray


def read_camera_infos(mission_dir, cameras=CAMERAS) -> dict[str, CamInfo]:
    bag = next(pathlib.Path(mission_dir).glob("*_alphasense.bag"))
    topic2cam = {f"/boxi/alphasense/{c}/camera_info": c for c in cameras}
    ts = get_typestore(Stores.ROS1_NOETIC)
    out: dict[str, CamInfo] = {}
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic in topic2cam]
        for conn, _, raw in reader.messages(conns):
            cam = topic2cam[conn.topic]
            if cam in out:
                continue
            m: Any = ts.deserialize_ros1(raw, conn.msgtype)
            if m.distortion_model != "equidistant":
                raise ValueError(f"{cam}: expected equidistant, got {m.distortion_model}")
            out[cam] = CamInfo(
                K=np.asarray(m.K, np.float64).reshape(3, 3),
                D=np.asarray(m.D, np.float64).reshape(-1)[:4].reshape(4, 1),
                size=(int(m.width), int(m.height)),
            )
            if len(out) == len(cameras):
                break
    missing = set(cameras) - set(out)
    if missing:
        raise ValueError(f"camera_info missing for {missing} in {bag}")
    return out


def _stereo_maps(ref: CamInfo, partner: CamInfo, t_partner_ref: np.ndarray,
                 balance: float, fov_scale: float):
    """Stereo-rectify ref (cam1, reference) and partner (cam2)."""
    w, h = ref.size
    R = t_partner_ref[:3, :3]
    T = t_partner_ref[:3, 3]
    R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
        ref.K, ref.D, partner.K, partner.D, (w, h), R, T,
        cv2.fisheye.CALIB_ZERO_DISPARITY, (0, 0), balance, fov_scale,
    )
    k_ref = P1[:3, :3].copy()
    k_partner = P2[:3, :3].copy()
    m1 = cv2.fisheye.initUndistortRectifyMap(ref.K, ref.D, R1, k_ref, (w, h), cv2.CV_16SC2)
    m2 = cv2.fisheye.initUndistortRectifyMap(partner.K, partner.D, R2, k_partner, (w, h), cv2.CV_16SC2)
    baseline = float(-P2[0, 3] / P2[0, 0])
    return Maps(m1[0], m1[1], k_ref), Maps(m2[0], m2[1], k_partner), R1, R2, baseline


def build_rectifiers(mission_dir, balance=DEFAULT_BALANCE, fov_scale=DEFAULT_FOV_SCALE) -> Rectifiers:
    infos = read_camera_infos(mission_dir)
    t_base_ref = get_static_transform(mission_dir, "base", FRAME[REF_CAM]).as_matrix()
    t_base_partner = get_static_transform(mission_dir, "base", FRAME[PARTNER_CAM]).as_matrix()
    t_partner_ref = inv(t_base_partner) @ t_base_ref

    ref, partner, R_ref, R_partner, baseline = _stereo_maps(
        infos[REF_CAM], infos[PARTNER_CAM], t_partner_ref, balance, fov_scale)
    return Rectifiers(ref, partner, baseline, R_ref, R_partner, t_base_ref, t_base_partner)


def remap(img: np.ndarray, maps: Maps) -> np.ndarray:
    return cv2.remap(img, maps.map1, maps.map2,
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
