"""Frozen DINO feature extraction."""

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dino(name="dinov2_vitb14_reg", weights=None, device="cuda"):
    if "/" in name:  # HF Hub id, e.g. facebook/dinov3-vitb16-pretrain-lvd1689m
        from transformers import AutoModel

        model = AutoModel.from_pretrained(name)
    else:
        repo = "facebookresearch/dinov3" if name.startswith("dinov3") else "facebookresearch/dinov2"
        kwargs = {"weights": weights} if weights else {}
        model = torch.hub.load(repo, name, **kwargs)
    return model.eval().to(device)


def patch_size(model):
    if hasattr(model, "config"):
        return model.config.patch_size
    return model.patch_size if hasattr(model, "patch_size") else model.patch_embed.patch_size[0]


@torch.no_grad()
def dino_patch_features(model, images, patch):
    """Last-layer DINO patch features for a batch of images."""
    device = images.device
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    H, W = images.shape[-2:]
    Hc, Wc = H // patch * patch, W // patch * patch
    x = images if (Hc, Wc) == (H, W) else F.interpolate(images, size=(Hc, Wc), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
        if hasattr(model, "config"):
            out = model(pixel_values=x).last_hidden_state.float()
            tokens = out[:, 1 + model.config.num_register_tokens :]  # (B, h*w, D)
            h, w = Hc // patch, Wc // patch
            return tokens.view(tokens.shape[0], h, w, -1).permute(0, 3, 1, 2)
        return model.get_intermediate_layers(x, n=1, reshape=True)[0].float()  # (B, D, h, w)


@torch.no_grad()
def dino_pooled_features(model, images, patch):
    """Mean-pooled DINO patch features for a batch of images."""
    return dino_patch_features(model, images, patch).mean(dim=(2, 3))
