"""Load a trained TrainingModule checkpoint for inference/feature extraction."""

import torch
from traversability_network.config import get_cfg
from traversability_network.network.trainer import TrainingModule


def load_checkpoint(cfg_file, checkpoint_path, device="cuda"):
    """Same pattern as visualize_predictions.load_model, plus COMPILE=False: eager
    mode is what lets BEVFeatureHook see a plain nn.Module, and inference here runs
    with varying batch sizes/missions where torch.compile would just recompile."""
    configs = get_cfg(cfg_file)
    configs.TRAINING.COMPILE = False
    model = TrainingModule(configs)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print(f"non-strict load: missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    return configs, model


class BEVFeatureHook:
    """Captures TravNet's penultimate BEV feature map (B, latent_dim, X, Y), the
    input to travmap_head, on every forward call without touching TravNet itself."""

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
