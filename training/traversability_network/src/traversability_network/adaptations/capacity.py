"""Reduced-capacity BEV stack (MODEL.SLIM_DECODER): 1x1 BEV compressor, no deepest decoder stage."""

import torch.nn as nn


def bev_compressor(in_channels, out_channels, slim):
    kernel = 1 if slim else 3
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))


def bottleneck(layer3, up3_skip, slim):
    """(deepest decoder stage, its skip upsample), both dropped when slim."""
    return (None, None) if slim else (layer3, up3_skip)


def run_bottleneck(layer3, up3_skip, x):
    """Down to H/8 and back up, or straight through when the stage is dropped."""
    return x if layer3 is None else up3_skip(layer3(x), x)
