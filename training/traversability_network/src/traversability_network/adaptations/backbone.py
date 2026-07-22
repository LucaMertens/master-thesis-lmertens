"""Image-backbone selection: ResNet or Dino"""
from ..network.models.encoder_decoder import Encoder
from .dino_neck import DinoNeck


def build_encoder(backbone, out_channels, downsample, input_size, freeze,
                  dino_feat_dim, dino_neck_dim):
    """Return the image encoder. ``input_size`` is (H, W). ``out_channels`` is the
    combined depth-bins + camera-feature channel count (self.D + self.camC)."""
    if backbone == "dino":
        out_hw = (-(-input_size[0] // downsample), -(-input_size[1] // downsample))
        return DinoNeck(out_channels, feat_dim=dino_feat_dim, mid_dim=dino_neck_dim,
                        out_hw=out_hw)
    return Encoder(out_channels, downsample=downsample, freeze=freeze)
