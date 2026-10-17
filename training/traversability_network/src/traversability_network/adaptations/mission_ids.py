"""Per-sample mission index and the order of optional trailing sample elements."""

import torch

# Trailing sample order: DINO features, mission index, human W-RIZZ pairs


def emit(rosbag_dict):
    return torch.tensor(rosbag_dict.get("mission_id", 0), dtype=torch.long)


def split(extra, uses_dino, uses_missions, uses_human=False):
    """(image_feats, mission_id, human) out of a batch's trailing elements."""
    i = 0
    feats = mission_id = human = None
    if uses_dino and len(extra) > i:
        feats = extra[i]
        i += 1
    if uses_missions and len(extra) > i:
        mission_id = extra[i]
        i += 1
    if uses_human and len(extra) > i:
        human = extra[i]
    return feats, mission_id, human


def flags(configs):
    """(uses_dino, uses_missions, uses_human) for the split above."""
    return (
        getattr(configs.MODEL, "BACKBONE", "resnet34") == "dino",
        getattr(configs.DATASET, "NAME", "grandtour") != "wayfaster",
        getattr(configs.TRAINING, "HUMAN_WEIGHT", 0.0) > 0,
    )
