"""Frozen DINOv2/v3 feature extraction, reused from side_experiments/dino_mu_probe
but pooled per-frame instead of point-sampled (OOD scoring works at window/frame
granularity, not path-cell granularity)."""

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dino(name="dinov2_vitb14_reg", weights=None, device="cuda"):
    repo = "facebookresearch/dinov3" if name.startswith("dinov3") else "facebookresearch/dinov2"
    kwargs = {"weights": weights} if weights else {}
    model = torch.hub.load(repo, name, **kwargs)
    return model.eval().to(device)


def patch_size(model):
    return model.patch_size if hasattr(model, "patch_size") else model.patch_embed.patch_size[0]


@torch.no_grad()
def dino_pooled_features(model, images, patch):
    """images: (B, 3, H, W) in [0, 1] RGB. Returns (B, D) mean-pooled last-layer
    patch features, D depending on the loaded DINO variant."""
    device = images.device
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    H, W = images.shape[-2:]
    Hc, Wc = H // patch * patch, W // patch * patch
    x = images if (Hc, Wc) == (H, W) else F.interpolate(images, size=(Hc, Wc), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
        fmap = model.get_intermediate_layers(x, n=1, reshape=True)[0].float()  # (B, D, h, w)
    return fmap.mean(dim=(2, 3))
