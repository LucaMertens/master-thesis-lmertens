"""Dropout injection for the BEV traversability head."""

import torch.nn as nn


def head_dropout(p):
    """Dropout2d layers for the trav head, empty when disabled (channel-wise, as neighbouring cells are correlated)."""
    return [nn.Dropout2d(p)] if p and p > 0 else []
