"""Small trainable neck over frozen (precomputed) DINO patch features."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoNeck(nn.Module):
    """Maps (B*T, feat_dim, h, w) cached DINO patch features to the stride-8
    encoder output (B*T, C, out_hw) holding depth logits + context channels.

    Input is L2-normalized per patch vector (over the channel dim). DINOv3 patch features carry the low/high-mu signal in a component
    that mean-centering (LayerNorm/GroupNorm) or per-dim standardization destroys
    (probe AUROC drops from 0.85 to ~0.55), while L2-normalization preserves it.
    A learnable per-channel gain follows, so the projection sees a stable scale.

    Submodule names must not contain "backbone" or "layer" (the trainer's
    optimizer groups match those to find pretrained params)."""

    def __init__(self, C, feat_dim=768, mid_dim=256, out_hw=(45, 60)):
        super().__init__()
        self.C = C
        self.out_hw = tuple(out_hw)
        self.gain = nn.Parameter(torch.ones(1, feat_dim, 1, 1))
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, mid_dim, kernel_size=1),
            nn.GELU())
        self.conv = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True))
        self.head = nn.Conv2d(mid_dim, C, kernel_size=1)

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1) * self.gain
        x = self.proj(x)
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.head(self.conv(x))
