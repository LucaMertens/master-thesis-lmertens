"""Warm-start weight loading for MODEL.LOAD_NETWORK, and config repair for old checkpoints."""

import torch

# Config-derived constants stored in the state_dict. Shapes always match, so they
# would silently overwrite this run's geometry.
CONST_LEAVES = {"voxels", "dx", "bx", "nx", "mean", "std", "mu_centers"}


def fill_defaults(configs):
    """Add config keys that did not exist when the checkpoint was written."""
    from ..config import get_cfg

    def merge(defaults, stored):
        for key, val in defaults.items():
            if key not in stored:
                stored[key] = val
            elif isinstance(val, type(stored)) and hasattr(val, "items"):
                merge(val, stored[key])
        return stored

    return merge(get_cfg(None), configs)


def load_pretrained(model, path):
    # weights_only=False: checkpoints carry EasyDict hparams
    pretrained = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

    # match torch.compile's "._orig_mod" key prefix to this run
    pretrained = {k.replace("._orig_mod", ""): v for k, v in pretrained.items()}
    own = model.state_dict()
    if any("._orig_mod" in k for k in own):
        pretrained = {
            k.replace("model.", "model._orig_mod.", 1) if k.startswith("model.") else k: v
            for k, v in pretrained.items()
        }

    # geometry and norm constants stay at this run's config values
    stale = []
    for k in [k for k in pretrained if k.rsplit(".", 1)[-1] in CONST_LEAVES]:
        if k not in own:
            del pretrained[k]
        elif not torch.equal(own[k].cpu(), pretrained[k].cpu()):
            stale.append(k)
            pretrained[k] = own[k]
        else:
            pretrained[k] = own[k]
    if stale:
        print(f"  checkpoint constants overridden by this run's config: {stale}")

    # head changes: drop shape-mismatched keys (load_state_dict raises on them even if non-strict)
    mismatched = [k for k, v in pretrained.items() if k in own and own[k].shape != v.shape]
    for k in mismatched:
        del pretrained[k]
    if mismatched:
        print(f"  dropped {len(mismatched)} shape-mismatched keys: {mismatched}")

    missing, unexpected = model.load_state_dict(pretrained, strict=False)
    if missing or unexpected:
        print(f"  non-strict load: missing={missing} unexpected={unexpected}")
