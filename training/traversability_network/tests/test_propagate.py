"""Geometric propagation: synthetic pose algebra plus a real-mission round trip."""

import os

import numpy as np
import pytest

from traversability_network.adaptations import propagate as P

MISSION = "/datahome/lmertens/grand_tour_dataset/extracted/ALB-3"
DEPTH_ROOT = "/datalocal/lmertens/grand_tour_dataset_depth"
needs_data = pytest.mark.skipif(not os.path.isdir(MISSION), reason="mission data absent")


def test_border_crop_matches_dataset():
    from traversability_network.adaptations.grandtour_dataset import BORDER_CROP

    assert P.BORDER_CROP == BORDER_CROP


class FakeGeom:
    """Two cameras looking down the world x axis, separated along y."""

    def __init__(self, n=101, baseline=0.05, depth=None):
        self.K = np.array([[500.0, 0, 720.0], [0, 500.0, 540.0], [0, 0, 1.0]])
        self.cam2base = np.eye(4)
        self.ts_ns = np.arange(n, dtype=np.int64)
        self.T_world_cam = np.tile(np.eye(4), (n, 1, 1))
        self.T_world_cam[:, 1, 3] = baseline * np.arange(n)
        self.T_cam_world = np.linalg.inv(self.T_world_cam)
        self.ground_z = np.zeros(n)
        self._depth = depth

    def __len__(self):
        return len(self.ts_ns)

    def depth(self, i):
        return self._depth


def unproject(geom, u, v, z):
    ray = np.array([(u - geom.K[0, 2]) / geom.K[0, 0], (v - geom.K[1, 2]) / geom.K[1, 1], 1.0])
    return (geom.T_world_cam[50] @ np.append(ray * z, 1.0))[:3]


def test_reprojects_into_own_frame_exactly():
    geom = FakeGeom()
    u, v, z = 800.0, 600.0, 3.0
    p = unproject(geom, u, v, z)
    cam = geom.T_cam_world[50] @ np.append(p, 1.0)
    assert cam[2] == pytest.approx(z, abs=1e-9)
    assert geom.K[0, 0] * cam[0] / cam[2] + geom.K[0, 2] == pytest.approx(u, abs=1e-4)
    assert geom.K[1, 1] * cam[1] / cam[2] + geom.K[1, 2] == pytest.approx(v, abs=1e-4)


def test_translation_moves_pixel_by_analytic_parallax():
    geom = FakeGeom(baseline=0.05)
    u, v, z = 720.0, 540.0, 2.0
    p = unproject(geom, u, v, z)
    cam = geom.T_cam_world[60] @ np.append(p, 1.0)
    # camera moved +0.5 m along world y, so the point moves -0.5 m in cam y
    assert cam[1] == pytest.approx(-0.5, abs=1e-9)
    assert cam[0] == pytest.approx(0.0, abs=1e-9)
    assert geom.K[1, 1] * cam[1] / cam[2] + geom.K[1, 2] == pytest.approx(v - 500 * 0.5 / z, abs=1e-4)
    assert geom.K[0, 0] * cam[0] / cam[2] + geom.K[0, 2] == pytest.approx(u, abs=1e-4)


def test_residual_gate_rejects_point_displaced_along_ray():
    z = 3.0
    depth = np.full((1080, 1440), int(z * 1000), np.uint16)  # mm
    geom = FakeGeom(depth=depth)
    p_ok = unproject(geom, 800.0, 600.0, z)
    p_far = unproject(geom, 800.0, 600.0, z + 1.0)  # behind the measured surface
    n_ok = len(P.propagate_point(p_ok, 50, geom)[0])
    n_far = len(P.propagate_point(p_far, 50, geom)[0])
    assert n_ok > 0
    assert n_far == 0


def test_one_sided_support_is_dropped():
    geom = FakeGeom(depth=np.full((1080, 1440), 3000, np.uint16))
    p = unproject(geom, 800.0, 600.0, 3.0)
    # no frames before the first anchor
    assert len(P.propagate_point(p, 0, geom)[0]) == 0


def test_thinning_respects_baseline_and_cap():
    geom = FakeGeom(baseline=0.05, depth=np.full((1080, 1440), 3000, np.uint16))
    p = unproject(geom, 800.0, 600.0, 3.0)
    frames, uv, z, resid = P.propagate_point(p, 50, geom, min_baseline=0.25, max_obs=24)
    assert len(frames) <= 24
    centers = geom.T_world_cam[frames, :3, 3]
    gaps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    assert (gaps >= 0.25 - 1e-9).all()


@needs_data
def test_static_point_holds_its_height_across_frames():
    """A static ground point must stay at ground height from other frames (checks the pose chain)."""
    geom = P.MissionGeometry(MISSION, DEPTH_ROOT)
    T_base_world = np.linalg.inv(geom.T_world_base)
    rng = np.random.default_rng(0)
    drift = {25: [], 50: []}
    for _ in range(300):
        f = int(rng.integers(200, len(geom) - 200))
        u, v = int(rng.integers(300, 1140)), int(rng.integers(650, 950))
        a = P.anchor_from_pixel((u, v), f, geom, mode="ground_plane")
        if a["status"] != "ok":
            continue
        p = np.append(a["p_world"], 1.0)
        for k in drift:
            for j in (f - k, f + k):
                h = (T_base_world[j] @ p)[2] - geom.ground_z[j]
                drift[k].append(abs(h - a["height_over_ground"]))
    assert len(drift[50]) >= 200
    assert np.median(drift[25]) < 0.15, f"drift at +/-25: {np.median(drift[25]):.3f} m"
    assert np.median(drift[50]) < 0.25, f"drift at +/-50: {np.median(drift[50]):.3f} m"


@needs_data
def test_depth_ground_clicks_are_near_the_ground():
    """Loose check: clicks in the lower image half are mostly ground."""
    geom = P.MissionGeometry(MISSION, DEPTH_ROOT)
    rng = np.random.default_rng(0)
    heights = []
    for _ in range(600):
        f = int(rng.integers(200, len(geom) - 200))
        u, v = int(rng.integers(200, 1240)), int(rng.integers(700, 1000))
        a = P.anchor_from_pixel((u, v), f, geom, mode="depth")
        if a["status"] == "ok":
            heights.append(a["height_over_ground"])
        if len(heights) >= 200:
            break
    assert len(heights) >= 100, f"only {len(heights)} valid anchors"
    assert abs(np.median(heights)) < 0.25, f"median height {np.median(heights):.3f} m"


@needs_data
def test_propagation_yield_and_residuals():
    geom = P.MissionGeometry(MISSION, DEPTH_ROOT)
    rng = np.random.default_rng(1)
    counts = []
    for _ in range(30):
        f = int(rng.integers(200, len(geom) - 200))
        u, v = int(rng.integers(300, 1140)), int(rng.integers(700, 1000))
        a = P.anchor_from_pixel((u, v), f, geom, mode="depth")
        if a["status"] != "ok":
            continue
        frames, uv, z, resid = P.propagate_point(a["p_world"], f, geom)
        if len(frames):
            counts.append(len(frames))
            assert (resid <= np.maximum(P.RESID_ABS, P.RESID_FRAC * z) + 1e-6).all()
            assert (frames >= f - P.WINDOW).all() and (frames <= f + P.WINDOW).all()
    assert len(counts) >= 10, f"only {len(counts)} anchors propagated"
    assert 5 <= np.median(counts) <= P.MAX_OBS
