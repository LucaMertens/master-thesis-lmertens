"""Image-backbone selection: ResNet or Dino"""
from .dino_neck import DinoNeck


def build_encoder(backbone, out_channels, downsample, input_size, freeze,
                  dino_feat_dim, dino_neck_dim, dino_noise=0.0):
    """Return the image encoder. ``input_size`` is (H, W)."""
    # local import: the network package imports this module back
    from ..network.models.encoder_decoder import Encoder

    if backbone == "dino":
        out_hw = (-(-input_size[0] // downsample), -(-input_size[1] // downsample))
        return DinoNeck(out_channels, feat_dim=dino_feat_dim, mid_dim=dino_neck_dim,
                        out_hw=out_hw, noise=dino_noise)
    # ADAPTATION: resnet18 is the encoder W-RIZZ (and the original TravNet) uses
    return Encoder(out_channels, downsample=downsample, freeze=freeze,
                   arch=18 if backbone == "resnet18" else 34)
