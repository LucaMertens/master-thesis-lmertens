"""Viewpoint disagreement: the frame composition has to be right or the score is noise."""

import numpy as np
import pytest
import torch

from ood_detection.common.viewpoint import disagreement, partner_index, to_frame

BOUNDS = {"xbound": [-2.0, 8.0, 0.1], "ybound": [-5.0, 5.0, 0.1]}


@pytest.fixture
def centers():
    x = np.arange(*BOUNDS["xbound"])[::-1]
    y = np.arange(*BOUNDS["ybound"])[::-1]
    return np.stack(np.meshgrid(x, y, indexing="ij"), -1).astype(np.float32)


def world_field(centers, pose):
    """mu as a function of world x only, seen from `pose`."""
    yaw = pose[2]
    wx = np.cos(yaw) * centers[..., 0] - np.sin(yaw) * centers[..., 1] + pose[0]
    return np.clip(wx / 20 + 0.5, 0, 1).astype(np.float32)


def roundtrip(v, src, dst):
    return to_frame(*(torch.as_tensor(a, dtype=torch.float32) for a in (v, src, dst)))


def test_to_frame_is_the_inverse_of_itself():
    xy = np.random.default_rng(0).normal(size=(1, 7, 2)).astype(np.float32)
    a, b = np.array([[1.0, -2.0, 0.4]]), np.array([[-3.0, 0.5, -1.1]])
    assert np.allclose(roundtrip(roundtrip(xy, a, b), b, a).numpy(), xy, atol=1e-4)


def test_partner_index_stays_inside_a_mission():
    m = np.array(["A", "A", "A", "B", "B"])
    assert list(partner_index(m, 2)) == [2, -1, -1, -1, -1]


def test_world_consistent_field_scores_near_zero(centers):
    poses = np.array([[0.0, 0.0, 0.0], [1.5, 0.4, 0.2], [3.0, 0.8, 0.35]])
    mu = np.stack([world_field(centers, p) for p in poses])
    obs = np.ones(mu.shape, bool)
    s = disagreement(mu, obs, poses, np.array(["M"] * 3), centers, BOUNDS, k=1)
    assert np.nanmax(s) < 0.02


def test_frame_locked_field_disagrees_under_rotation(centers):
    poses = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.6]])
    mu = np.stack([np.clip(centers[..., 0] / 20 + 0.5, 0, 1).astype(np.float32)] * 2)
    s = disagreement(mu, np.ones(mu.shape, bool), poses, np.array(["M"] * 2), centers, BOUNDS, k=1)
    assert s[0] > 0.05


def test_offset_prediction_shows_up_as_the_offset(centers):
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mu = np.stack([world_field(centers, p) for p in poses])
    mu[1] += 0.3
    s = disagreement(mu, np.ones(mu.shape, bool), poses, np.array(["M"] * 2), centers, BOUNDS, k=1)
    assert abs(s[0] - 0.3) < 0.02


def test_unobserved_cells_do_not_count(centers):
    poses = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    mu = np.stack([world_field(centers, p) for p in poses])
    mu[1, :50] = 0.0  # garbage, but masked out
    obs = np.ones(mu.shape, bool)
    obs[1, :50] = False
    s = disagreement(mu, obs, poses, np.array(["M"] * 2), centers, BOUNDS, k=1)
    assert np.nanmax(s) < 0.02


def test_no_partner_is_nan(centers):
    poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mu = np.stack([world_field(centers, p) for p in poses])
    s = disagreement(mu, np.ones(mu.shape, bool), poses, np.array(["A", "B"]), centers, BOUNDS, k=1)
    assert np.isnan(s).all()
