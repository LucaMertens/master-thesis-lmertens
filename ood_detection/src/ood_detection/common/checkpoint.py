"""Checkpoint loading and forward hooks for feature extraction."""

import torch
from traversability_network.config import get_cfg
from traversability_network.network.trainer import TrainingModule


def load_checkpoint(cfg_file, checkpoint_path, device="cuda"):
    configs = get_cfg(cfg_file)
    configs.TRAINING.COMPILE = False
    model = TrainingModule(configs)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # compiled checkpoints prefix keys with "._orig_mod"
    state_dict = {k.replace("._orig_mod", ""): v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"non-strict load: missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    return configs, model


class BEVFeatureHook:
    """Captures TravNet's penultimate BEV feature map on every forward call."""

    def __init__(self, model):
        self.features = None
        self.handle = model.model.decoder.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.features = out

    def remove(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


class DepthLogitsHook:
    """Captures the depth head logits (first D encoder channels). Needs MODEL.PREDICT_DEPTH."""

    def __init__(self, model):
        self.logits = None
        self.n_bins = model.model.D
        self.handle = model.model.encoder.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.logits = out[:, :self.n_bins]

    def entropy(self, batch, frames):
        """(batch,) mean predictive entropy over the last input frame's pixels."""
        import torch

        logp = torch.log_softmax(self.logits.float(), 1)
        h = -(logp.exp() * logp).sum(1).mean((-2, -1))
        return h.view(batch, frames)[:, -1]

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def image_context(model, color, image_feats=None):
    """(B*T, camC, H/ds, W/ds) context channels of the encoder, without the depth channels."""
    net = model.model
    B, T = color.shape[:2]
    if image_feats is not None:
        out = net.encoder(image_feats.view(B * T, *image_feats.shape[2:]).float())
    else:
        out = net.encoder((color.view(B * T, *color.shape[2:]) - net.mean) / net.std)
    return out[:, net.D:net.D + net.camC]
