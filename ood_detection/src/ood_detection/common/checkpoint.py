"""Load a trained TrainingModule checkpoint for inference/feature extraction."""

import torch
from traversability_network.config import get_cfg
from traversability_network.network.trainer import TrainingModule


def load_checkpoint(cfg_file, checkpoint_path, device="cuda"):
    """Loads a checkpoint into an eager (uncompiled) model for inference."""
    configs = get_cfg(cfg_file)
    configs.TRAINING.COMPILE = False
    model = TrainingModule(configs)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # compiled-model checkpoints prefix keys with "._orig_mod"; strip it for the eager model
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
