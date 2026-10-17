"""Per-dataset mu calibration for mixed-dataset training (see TRAINING.PER_DATASET_AFFINE)."""

import torch
from torch import nn

GRANDTOUR_ID = 0
FORENT_ID = 1
N_DATASETS = 2


class PerDatasetAffine(nn.Module):
    """mu' = clamp(exp(log_scale) * mu + offset, 0, 1), learned per dataset.

    Dataset 0 (GrandTour) stays identity so metrics remain comparable with single-dataset runs.
    """

    def __init__(self, n_datasets=N_DATASETS):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(n_datasets - 1))
        self.offset = nn.Parameter(torch.zeros(n_datasets - 1))

    def forward(self, mu, dataset_id):
        ones = mu.new_ones(1)
        zeros = mu.new_zeros(1)
        scale = torch.cat([ones, self.log_scale.exp()])[dataset_id]
        shift = torch.cat([zeros, self.offset])[dataset_id]
        view = (-1,) + (1,) * (mu.dim() - 1)
        return (scale.view(view) * mu + shift.view(view)).clamp(0.0, 1.0)

    def log(self, pl_module):
        for i, (s, o) in enumerate(zip(self.log_scale.exp(), self.offset), start=1):
            pl_module.log(f"affine/scale_{i}", s)
            pl_module.log(f"affine/offset_{i}", o)


def apply(affine, trav_map, dataset_id):
    """Calibrate the mu channel of trav_map."""
    if affine is None or dataset_id is None:
        return trav_map
    mu = affine(trav_map[:, 0], dataset_id)
    return torch.cat([mu.unsqueeze(1), trav_map[:, 1:]], dim=1)
