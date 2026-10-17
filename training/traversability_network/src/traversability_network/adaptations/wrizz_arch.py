"""W-RIZZ decoder variant: nearest upsampling plus 3x3 convs instead of bilinear plus 1x1.

The ResNet-18 encoder and RGB-only input are config (MODEL.BACKBONE, MODEL.FUSE_PCLOUD).
"""

import torch.nn as nn
import torch.nn.functional as F


class UpsamplingAddNearest(nn.Module):
    """Drop-in for encoder_decoder.UpsamplingAdd with the paper's upsampling."""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x, x_skip):
        # odd sizes overshoot the skip by one (13 -> 26 vs 25); F.pad crops on negative diff
        x = self.upsample_layer(x)
        dy = x_skip.size(2) - x.size(2)
        dx = x_skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return x + x_skip


def apply(decoder):
    """Swap every UpsamplingAdd in a Decoder for the nearest/3x3 variant, in place."""
    for name, module in list(decoder.named_children()):
        if type(module).__name__ == "UpsamplingAdd":
            layer = module.upsample_layer
            conv = next(m for m in layer if isinstance(m, nn.Conv2d))
            scale = next(m for m in layer if isinstance(m, nn.Upsample)).scale_factor
            setattr(decoder, name,
                    UpsamplingAddNearest(conv.in_channels, conv.out_channels, scale))
    return decoder
