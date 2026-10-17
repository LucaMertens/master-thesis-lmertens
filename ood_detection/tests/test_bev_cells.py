import numpy as np
import torch

from ood_detection.common.bev_cells import ground_height, sample_cells, visible_cells

GRID = {"xbound": [-2.0, 8.0, 0.1], "ybound": [-5.0, 5.0, 0.1],
        "zbound": [-1.5, 1.5, 0.2], "dbound": [0.3, 8.0, 0.2]}
IMAGE = [480, 360]


def forward_camera():
    """Camera at the base origin looking along +x, with cam = (p - t) @ R."""
    ext = torch.eye(4)
    ext[:3, :3] = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    K = torch.tensor([[300.0, 0, 240.0], [0, 300.0, 180.0], [0, 0, 1.0]])
    return K.view(1, 1, 3, 3), ext.view(1, 1, 4, 4)


def flat_ground(shape):
    return torch.zeros((1,) + shape)


def test_only_cells_in_front_of_the_camera_are_observed():
    K, ext = forward_camera()
    centers = torch.tensor([[[2.0, 0.0], [-2.0, 0.0]], [[4.0, 0.0], [2.0, 4.0]]])
    seen = visible_cells(centers, flat_ground((2, 2)), K, ext, GRID, IMAGE)
    assert seen.tolist() == [[[True, False], [True, False]]]


def test_cells_beyond_dbound_are_dropped():
    K, ext = forward_camera()
    centers = torch.tensor([[[0.2, 0.0], [9.0, 0.0]]])
    seen = visible_cells(centers, flat_ground((1, 2)), K, ext, GRID, IMAGE)
    assert seen.tolist() == [[[False, False]]]


def test_occlusion_hides_a_cell_behind_a_closer_surface():
    K, ext = forward_camera()
    centers = torch.tensor([[[4.0, 0.0]]])
    depth = torch.full((1, 1, 360, 480), 1000, dtype=torch.int16)  # a wall at 1 m
    assert visible_cells(centers, flat_ground((1, 1)), K, ext, GRID, IMAGE).item()
    assert not visible_cells(centers, flat_ground((1, 1)), K, ext, GRID, IMAGE, depth).item()


def test_ground_height_follows_a_sloped_path():
    """z = -0.5 under the robot, dropping 0.1 m per metre of x."""
    states = torch.full((1, 40, 6), float("inf"))
    x = torch.linspace(0.0, 6.0, 30)
    states[0, :30, 0] = x
    states[0, :30, 1] = 0.0
    states[0, :30, 5] = -0.5 - 0.1 * x
    centers = torch.tensor([[[0.0, 0.0], [4.0, 0.0]]])
    z = ground_height(states, centers, GRID)
    assert torch.allclose(z, torch.tensor([[[-0.5, -0.9]]]), atol=1e-3)


def test_ground_height_falls_back_to_the_median_without_enough_states():
    states = torch.full((1, 40, 6), float("inf"))
    states[0, :3, :2] = 0.0
    states[0, :3, 5] = torch.tensor([-0.5, -0.6, -0.4])
    centers = torch.tensor([[[0.0, 0.0], [4.0, 0.0]]])
    assert torch.allclose(ground_height(states, centers, GRID), torch.full((1, 1, 2), -0.5))


def test_sample_cells_draws_only_observed_cells_when_enough_exist():
    features = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask[:, :2] = True
    cells, n = sample_cells(features, mask, 4)
    assert n.tolist() == [8, 8]
    observed = features.reshape(2, 3, -1).transpose(1, 2)[:, :8]
    for b in range(2):
        drawn = {tuple(c) for c in cells[b].tolist()}
        assert drawn <= {tuple(c) for c in observed[b].tolist()}
        assert len(drawn) == 4


def test_sample_cells_pads_when_too_few_are_observed():
    features = torch.randn(1, 3, 4, 4)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[0, 0, 0] = True
    cells, n = sample_cells(features, mask, 4)
    assert n.item() == 1
    assert cells.shape == (1, 4, 3)
    assert np.allclose(cells[0, 0].numpy(), features[0, :, 0, 0].numpy())
