"""Writes one config per regularization-grid arm (chapters/eval_grid.typ).

    uv run python training/traversability_network/configs/make_eval_grid.py
"""

import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "grandtour_baseline_f2.yaml")
PRETRAIN = "/datahome/lmertens/checkpoints/yqd58a88/checkpoint_epoch=19-step=126800.ckpt"

# Shared protocol: 10 epochs, 4 validations per epoch, weights-only checkpoint at each
PROTOCOL = {
    "TRAINING": {
        "DEVICES": [0],
        "WORKERS": 5,
        "EPOCHS": 10,
        "VAL_CHECK_INTERVAL": 0.25,
        "CHECKPOINT_EVERY_N_EPOCHS": 10,
        "CHECKPOINT_EVERY_N_VALIDATIONS": 1,
        "CKPT_MONITOR": "valid_mission/calib_slope",
        "CKPT_MODE": "max",
        "VIS_PANELS_SKIP": ["bev_outputs", "bev_label", "geom_frames", "images"],
    }
}

ARMS = {
    "gt_dino_backbone_f2": (
        "Backbone: frozen DINOv3 patch features and a small neck in place of the ImageNet "
        "ResNet34, decoder and baseline loss unchanged. The single-change arm the earlier "
        "DINO runs never were, since each of those also swapped the loss.",
        {"MODEL": {"BACKBONE": "dino",
                   "DINO_MODEL": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                   "DINO_FEAT_DIM": 768, "DINO_INPUT_SIZE": [480, 352],
                   "DINO_NECK_DIM": 256},
         "DATASET": {"DINO_CACHE_ROOT": "/datalocal/lmertens/grand_tour_dataset_dino"}},
    ),
    "gt_backbone_lr_10percent_f2": (
        "Backbone plasticity: ImageNet ResNet34 stages at 0.1x the head learning rate. "
        "Single change against gt_cv2_f2.",
        {"OPTIMIZER": {"BACKBONE_LR_FACTOR": 0.1}},
    ),
    "gt_backbone_frozen_f2": (
        "Backbone plasticity: ResNet34 stages frozen, so only the neck, BEV stack and "
        "heads train and depth has to be learned from fixed ImageNet features.",
        {"MODEL": {"FREEZE_BACKBONE": True}},
    ),
    "gt_decoder_slim_f2": (
        "Decoder capacity: bev_compressor as a 1x1 conv and the decoder's deepest stage "
        "dropped, 12.5 M to 9.9 M trainable parameters.",
        {"MODEL": {"SLIM_DECODER": True}},
    ),
    "gt_decoder_minimal_frozen_f2": (
        "Decoder capacity, smallest level: slim decoder on a frozen backbone with half "
        "the BEV width, 1.5 M trainable parameters.",
        {"MODEL": {"SLIM_DECODER": True, "FREEZE_BACKBONE": True, "LATENT_DIM": 32}},
    ),
    "gt_loss_gaussian_binned_f2": (
        "Loss function: mu as a 20-bin distribution trained against a Gaussian target "
        "(sigma 0.1) instead of two-hot, so a label is taught with its uncertainty.",
        {"TRAINING": {"LOSS_TYPE": "binned_ce", "MU_BIN_SIGMA": 0.1},
         "MODEL": {"MU_BINS": 20}},
    ),
    "gt_loss_focal_regression_f2": (
        "Loss function: Focal-R (Yang et al. 2021) on the mu regression term, "
        "down-weighting the cells the model already fits.",
        {"TRAINING": {"LOSS_TYPE": "focal_r", "FOCAL_BETA": 10.0, "FOCAL_GAMMA": 2.0}},
    ),
    "gt_weight_decay_0.01_f2": (
        "Weight decay 0.01, 100x the baseline, on the baseline loss.",
        {"OPTIMIZER": {"WEIGHT_DECAY": 0.01}},
    ),
    "gt_weight_decay_0.1_f2": (
        "Weight decay 0.1, the top of the tested range.",
        {"OPTIMIZER": {"WEIGHT_DECAY": 0.1}},
    ),
    "gt_depth_weight_0.5_f2": (
        "Depth weight 0.5 instead of 0.05: the depth head constrains the shared encoder "
        "ten times harder.",
        {"TRAINING": {"DEPTH_WEIGHT": 0.5}},
    ),
    "gt_wayfaster_pretrained_f2": (
        "Warm start from the WayFASTER-data run yqd58a88 (both cameras, epoch 19), "
        "whole network, baseline recipe otherwise.",
        {"MODEL": {"LOAD_NETWORK": PRETRAIN}},
    ),
    "gt_wayfaster_pretrained_frozen_f2": (
        "Warm start from WayFASTER run yqd58a88 with the encoder frozen, so its features "
        "replace ImageNet's instead of being fine-tuned away.",
        {"MODEL": {"LOAD_NETWORK": PRETRAIN, "FREEZE_BACKBONE": True}},
    ),
    "gt_dino_pixel_regression_f2": (
        "Decoder capacity past the minimal net: mu is a 1x1 linear layer on the frozen "
        "DINOv3 patch features lifted to BEV by the depth splat alone, with no mu "
        "decoder. Regression on the 20-bin expectation, LDS weighting as the baseline. "
        "The DINO backbone arms are its control: same features, decoder kept.",
        {"TRAINING": {"LOSS_TYPE": "regression"},
         "MODEL": {"BACKBONE": "dino",
                   "DINO_MODEL": "facebook/dinov3-vitb16-pretrain-lvd1689m",
                   "DINO_FEAT_DIM": 768, "DINO_INPUT_SIZE": [480, 352],
                   "DINO_NECK_DIM": 256, "MU_BINS": 20, "MU_PIXEL_LINEAR": True},
         "DATASET": {"DINO_CACHE_ROOT": "/datalocal/lmertens/grand_tour_dataset_dino"}},
    ),
}


def merge(base, delta):
    out = copy.deepcopy(base)
    for key, val in delta.items():
        if isinstance(val, dict):
            out[key] = merge(out.get(key, {}), val)
        else:
            out[key] = val
    return out


def main():
    with open(BASE) as fh:
        base = yaml.safe_load(fh)
    base = merge(base, PROTOCOL)
    for tag, (description, delta) in ARMS.items():
        cfg = merge(base, {"TAG": tag, "DESCRIPTION": description} | delta)
        path = os.path.join(HERE, f"{tag}.yaml")
        with open(path, "w") as fh:
            fh.write("# Generated by make_eval_grid.py, do not edit.\n")
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False, width=100)
        print("wrote", os.path.basename(path))


if __name__ == "__main__":
    main()
