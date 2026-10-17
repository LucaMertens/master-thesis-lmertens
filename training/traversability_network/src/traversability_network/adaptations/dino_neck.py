"""Small trainable neck over frozen (precomputed) DINO patch features."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoNeck(nn.Module):
    """Cached DINO patch features (B*T, feat_dim, h, w) -> encoder output (B*T, C, out_hw).

    L2 normalization per patch vector, not mean-centering or standardization: those
    destroy the low/high-mu signal (probe AUROC 0.85 -> ~0.55). ``noise`` perturbs on
    the unit sphere at train time only.

    Submodule names must not contain "backbone" or "layer" (trainer optimizer groups
    match those to find pretrained params)."""

    def __init__(self, C, feat_dim=768, mid_dim=256, out_hw=(45, 60), noise=0.0):
        super().__init__()
        self.C = C
        self.out_hw = tuple(out_hw)
        self.noise = noise
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
        x = F.normalize(x, p=2, dim=1)
        # stand-in for the colour jitter that cannot reach precomputed features
        if self.training and self.noise > 0:
            n = torch.randn_like(x) * (self.noise / x.shape[1] ** 0.5)
            x = F.normalize(x + n, p=2, dim=1)
        x = x * self.gain
        x = self.proj(x)
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.head(self.conv(x))
