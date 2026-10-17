# Taken from wayfaster repository (https://github.com/matval/wayfaster) train/train_configs.py
# Modified: GrandTour DATASET section, TIME_TOLERANCE / MAX_TIME_GAP, MODEL.LABEL_COLUMNS.
import copy
import sys

import yaml
from shared_utils.paths import (
    DATALOCAL_ROOT,
    EXTRACTED_ROOT,
    LABELS_ROOT,
    WRIZZ_INDEX_ROOT,
)
from easydict import EasyDict

_C = EasyDict()

_C.TAG = "grandtour"
# One-line summary of what the run tests; copied into the wandb run notes.
_C.DESCRIPTION = ""

_C.TRAINING = EasyDict()
_C.TRAINING.EPOCHS = 50
_C.TRAINING.BATCHSIZE = 4
_C.TRAINING.WORKERS = 8
_C.TRAINING.DEVICES = (
    -1
)  # GPUs for the Trainer; -1 = all visible, or an int/list to limit
_C.TRAINING.COMPILE = True  # torch.compile the model; set False to debug in eager
_C.TRAINING.PRECISION = "bf16-mixed"
# Fraction of an epoch between validations; periodic checkpoint interval
_C.TRAINING.VAL_CHECK_INTERVAL = 1.0
_C.TRAINING.CHECKPOINT_EVERY_N_EPOCHS = 5
# Archive a checkpoint every n-th validation (0 = off), every validation for the first DENSE_EPOCHS
_C.TRAINING.CHECKPOINT_EVERY_N_VALIDATIONS = 0
_C.TRAINING.CHECKPOINT_DENSE_EPOCHS = 0
# Metric the "best_*" checkpoint tracks; calibration slope falls when the output collapses to the mean
_C.TRAINING.CKPT_MONITOR = "valid/calib_slope"
_C.TRAINING.CKPT_MODE = "max"
# Gradient accumulation: effective batch = BATCHSIZE * GPUs * this.
_C.TRAINING.ACCUMULATE_GRAD_BATCHES = 1
# wandb run id to append to; None = whatever --resume resolves, else a fresh run
_C.TRAINING.WANDB_RESUME_ID = None
_C.TRAINING.DT = 0.1  # time step in seconds
_C.TRAINING.HORIZON = 300  # horizon in number of points
_C.TRAINING.GAMMA = 0.998
_C.TRAINING.DEPTH_WEIGHT = 0.1
# Extra penalty on optimistic errors (predicted trav > label)
_C.TRAINING.TRAV_OVERPRED_WEIGHT = 1.0
# smooth-L1 transition point (WayFASTER: 0.1); larger compresses toward the mean
_C.TRAINING.SMOOTH_L1_BETA = 0.1
# "weighted": divide by weight mass (adapted); "mean": plain mean (WayFASTER)
_C.TRAINING.LOSS_NORM = "weighted"
# Mu loss: None = "regression", "bmc", "binned_ce", "class3", "evidential", "rizz" or "focal_r"
_C.TRAINING.LOSS_TYPE = None
# Focal-R (Yang et al. 2021); beta is raised from the paper's 0.2 because mu spans 1, not ~100
_C.TRAINING.FOCAL_BETA = 10.0
_C.TRAINING.FOCAL_GAMMA = 2.0
# Gaussian target width for the binned mu head, in mu units (0 = two-hot)
_C.TRAINING.MU_BIN_SIGMA = 0.0
# W-RIZZ (Schreiber et al. 2024) ordinal mu supervision, LOSS_TYPE="rizz". The mu head
# then predicts a relative score; read valid/hdr and valid/auroc_low, not the confusion matrix.
# Squared-hinge margin, in output units
_C.TRAINING.RIZZ_MARGIN = 0.5
# Label mu gap below which two cells count as equally traversable (also HDR's threshold)
_C.TRAINING.RIZZ_DEADBAND = 0.25
# Largest mu gap still taught as a tie; gaps up to RIZZ_DEADBAND are dropped as ambiguous
_C.TRAINING.RIZZ_TIE_BAND = 0.125
# Pairs drawn per training batch, and per batch for the HDR metric.
_C.TRAINING.RIZZ_PAIRS = 512
_C.TRAINING.RIZZ_HDR_PAIRS = 2048
# Share of pairs whose cells come from different windows (paper: 1/3)
_C.TRAINING.RIZZ_CROSS_FRACTION = 0.5
# Minimum BEV separation of an intra-window pair, metres
_C.TRAINING.RIZZ_MIN_SEP = 2.0
# Target share of inequality pairs (random RHE pairs are almost all ties)
_C.TRAINING.RIZZ_INEQ_FRACTION = 0.5
# Inequality pairs intra-window only: cross-mission ones mostly teach recording identity
_C.TRAINING.RIZZ_INEQ_INTRA_ONLY = False
# Scale of the rizz term against the nu term
_C.TRAINING.RIZZ_WEIGHT = 1.0
# Weight on the RHE loss; 0 = RHE metrics only, no gradient (pure human-pair setting)
_C.TRAINING.TRAV_WEIGHT = 1.0
# Human W-RIZZ pair loss (side_experiments/wrizz_labeling), added to LOSS_TYPE; 0 = off
_C.TRAINING.HUMAN_WEIGHT = 0.0
# Only "bev" is implemented: scores the BEV cell under the annotated world point
_C.TRAINING.HUMAN_TARGET = "bev"
# Squared-hinge margin (paper: 0.5)
_C.TRAINING.HUMAN_MARGIN = 0.5
# Occurrences of one annotation per epoch, sharing one unit of loss mass (0 = every scoring window)
_C.TRAINING.HUMAN_K = 5
# Occurrences of a cross pair; each needs two co-resident windows
_C.TRAINING.HUMAN_K_CROSS = 1
# Mass share on the annotated window; the rest goes to near, middle and far views
_C.TRAINING.HUMAN_ANCHOR_MASS = 0.4
# Target mass share of inequality answers (0 = natural mix)
_C.TRAINING.HUMAN_INEQ_FRACTION = 0.5
# Compare every (point in A, point in B) instead of region means, averaged per annotation
_C.TRAINING.HUMAN_POINT_PAIRS = False
# "annotation": one unit of mass per answer; "combination": one per (A, B) cell pair
_C.TRAINING.HUMAN_POINT_MASS = "annotation"
# Cell pairs drawn per annotation per batch (0 = all combinations)
_C.TRAINING.HUMAN_POINT_SAMPLES = 0
# Mean teacher (Tarvainen & Valpola 2017): weight of the EMA-teacher consistency term, 0 = off
_C.TRAINING.MEAN_TEACHER_WEIGHT = 0.0
_C.TRAINING.MEAN_TEACHER_DECAY = 0.999
# Steps over which the consistency weight ramps in
_C.TRAINING.MEAN_TEACHER_RAMP = 2000
# Photometric jitter on the teacher view only (brightness, contrast, saturation, hue)
_C.TRAINING.MEAN_TEACHER_JITTER = [0.4, 0.4, 0.4, 0.1]
# class3 head: thresholded label mu, logit-adjusted CE (Menon et al. 2021); priors are train class frequencies
_C.TRAINING.MU_CLASS_THRESHOLDS = [0.3, 0.6]
_C.TRAINING.MU_CLASS_PRIORS = [0.0153, 0.026, 0.9587]
# Balanced MSE (BMC, Ren et al. 2022): initial learnable noise sigma, 0 = off
_C.TRAINING.BMC_SIGMA_INIT = 0.0
# Deep Evidential Regression (Amini et al. 2020) regularizer weight, used with MODEL.EVIDENTIAL
_C.TRAINING.EVIDENTIAL_LAMBDA = 0.01
_C.TRAINING.VIS_INTERVAL = 500
# Max images per visualized panel (batch elements are near-duplicate windows)
_C.TRAINING.VIS_BATCH_SIZE = 8
# Extra wandb debug panels (see TrainingModule.visualize)
_C.TRAINING.DEBUG_VIS_LABELS = False
_C.TRAINING.DEBUG_VIS_GEOM = False
_C.TRAINING.DEBUG_VIS_DEPTH = False
# wandb panels never logged or rendered, by short name
_C.TRAINING.VIS_PANELS_SKIP = []
_C.TRAINING.VERBOSE = True

# model parameters
_C.MODEL = EasyDict()
_C.MODEL.LOAD_NETWORK = None
# Full trainer state to resume from; normally None, --resume finds the TAG's last.ckpt
_C.MODEL.RESUME_CHECKPOINT = None
_C.MODEL.DOWNSAMPLE = 8
_C.MODEL.LATENT_DIM = 64
_C.MODEL.TIME_LENGTH = 6
_C.MODEL.PREDICT_DEPTH = True
_C.MODEL.TRAIN_DEPTH = True
_C.MODEL.FUSE_PCLOUD = True
# Nearest-subsample factor for the GPU voxelization (1 = full-res depth)
_C.MODEL.VOXEL_DEPTH_DOWNSAMPLE = 1
# (W, H) the voxelization depth is resampled to, K scaled to match; None = native
_C.MODEL.VOXEL_DEPTH_SIZE = None
_C.MODEL.INPUT_SIZE = [480, 360]  # (W, H); Alphasense native is 1440x1080 (4:3)
# Freeze the pretrained encoder stages (BN in eval); only neck and heads train
_C.MODEL.FREEZE_BACKBONE = False
# Dropout2d probability inside the BEV traversability head (0 = off).
_C.MODEL.HEAD_DROPOUT = 0.0
# Reduced BEV capacity, see adaptations/capacity.py
_C.MODEL.SLIM_DECODER = False
# "resnet34", "resnet18" or "dino" (frozen patch features from DATASET.DINO_CACHE_ROOT,
# L2-normalized in the DinoNeck; see side_experiments/dino_mu_probe)
_C.MODEL.BACKBONE = "resnet34"
# W-RIZZ's decoder (nearest upsampling + 3x3 conv) against checkerboarding
_C.MODEL.WRIZZ_DECODER = False
_C.MODEL.DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
_C.MODEL.DINO_FEAT_DIM = 768
# (W, H) fed to DINO at cache time, a patch-size multiple
_C.MODEL.DINO_INPUT_SIZE = [480, 352]
_C.MODEL.DINO_NECK_DIM = 256
# Train-time Gaussian noise on DINO features, stands in for color jitter
_C.MODEL.DINO_FEAT_NOISE = 0.0
# Mu head: 0 = sigmoid regression; K > 0 = K-bin categorical with two-hot CE
_C.MODEL.MU_BINS = 0
# Linear per-pixel mu probe on frozen DINO features lifted to BEV, no mu decoder
_C.MODEL.MU_PIXEL_LINEAR = False
# Evidential mu head (Normal-Inverse-Gamma); exclusive with MU_BINS and BMC_SIGMA_INIT
_C.MODEL.EVIDENTIAL = False
_C.MODEL.LABEL_COLUMNS = [0, 1]  # columns of rhe_trav_labels.npy used as (mu, nu)
_C.MODEL.GRID_BOUNDS = {
    "xbound": [-2.0, 8.0, 0.1],
    "ybound": [-5.0, 5.0, 0.1],
    "zbound": [-1.5, 1.5, 0.2],
    "dbound": [0.3, 8.0, 0.2],
}

# optimizer parameters
_C.OPTIMIZER = EasyDict()
_C.OPTIMIZER.LR = 1e-4
# Effective batch (BATCHSIZE * num GPUs) that LR was tuned at; LR is sqrt-scaled from here.
_C.OPTIMIZER.LR_REF_BATCHSIZE = 12
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4
# LR multiplier for the pretrained encoder stages
_C.OPTIMIZER.BACKBONE_LR_FACTOR = 1.0

_C.DATASET = EasyDict()
# "grandtour" or "wayfaster" (original CSV dataset, architecture control)
_C.DATASET.NAME = "grandtour"
_C.DATASET.CSV_FILE = "rosbags.csv"
# Data directories for NAME=wayfaster (each holds CSV_FILE + one dir per bag).
_C.DATASET.TRAIN_DIRS = []
_C.DATASET.VALID_DIRS = []
# Directory holding one sub-directory per mission (images, depth, odom, intrinsics, ...).
_C.DATASET.DATA_ROOT = str(EXTRACTED_ROOT)
# Directory holding one sub-directory per mission with rhe/ and simple/ label outputs.
_C.DATASET.LABELS_ROOT = str(LABELS_ROOT)
# Local-SSD root with pre-resized color images (resize_images.py); None = native res from DATA_ROOT
_C.DATASET.RESIZED_ROOT = str(DATALOCAL_ROOT / "grand_tour_dataset_resized")
_C.DATASET.DEPTH_CACHE_ROOT = str(DATALOCAL_ROOT / "grand_tour_dataset_depth")
# Per-mission index of human W-RIZZ annotations, only read when HUMAN_WEIGHT > 0
_C.DATASET.HUMAN_INDEX_ROOT = str(WRIZZ_INDEX_ROOT)
# Read raw .npy sidecars (adaptations/npy_cache.py), falls back to PNG per file
_C.DATASET.NPY_SIDECARS = True
# Per-mission frozen DINO patch-feature cache (features.npy + meta.npz), see MODEL.BACKBONE.
_C.DATASET.DINO_CACHE_ROOT = str(DATALOCAL_ROOT / "grand_tour_dataset_dino")
_C.DATASET.TRAIN_MISSIONS = []
_C.DATASET.VALID_MISSIONS = []
# Within-mission split: {mission: [t0, t1]} or {mission: [[t0, t1], ...]} in seconds.
# Missing missions are used whole; a cut mission appears in both TRAIN_ and VALID_MISSIONS.
_C.DATASET.TRAIN_TIME_RANGES = None
_C.DATASET.VALID_TIME_RANGES = None
# Derive both from shared_utils.tiers (tiers.json); mission lists are checked against it
_C.DATASET.TIER_TIME_RANGES = False
# Held-out ID test set and OOD-detection benchmark sets (consumed by eval, not training).
_C.DATASET.TEST_MISSIONS = []
_C.DATASET.NEAR_OOD_MISSIONS = []
_C.DATASET.FAR_OOD_MISSIONS = []
_C.DATASET.TIME_TOLERANCE = 0.05  # s, max camera<->RHE/depth timestamp mismatch
_C.DATASET.MAX_TIME_GAP = 0.5  # s, skip image sequences crossing a larger gap
# FRAME_STRIDE spaces the stacked frames (5 = 0.5 s at 10 Hz); WINDOW_STRIDE spaces window starts
_C.DATASET.FRAME_STRIDE = 1
_C.DATASET.WINDOW_STRIDE = 1
# ADAPTATION: FRAME_STRIDE per mission from its mean |v|, so windows span ~STRIDE_TARGET_M
# metres (default: WayFASTER's mean window extent). Falls back to FRAME_STRIDE at ~0 speed.
_C.DATASET.PER_MISSION_STRIDE = False
_C.DATASET.STRIDE_TARGET_M = 1.70
# Trim leading/trailing stationary frames whose whole label horizon has no motion.
_C.DATASET.TRIM_STATIONARY = True
_C.DATASET.MOTION_MIN_SPEED = 0.01  # m/s, |v| above this counts as moving
_C.DATASET.MOTION_MIN_OMEGA = 0.01  # rad/s, |omega| above this counts as moving
# Low-mu oversampling: replicate horizons with a cell (mu <= MAX, observability >= MIN) up to this fraction of the train index list
_C.DATASET.OVERSAMPLE_FRACTION = 0.40
# Data-amount ablation: deterministic fraction of train windows
_C.DATASET.TRAIN_FRACTION = 1.0
# Same ablation cut by whole recordings instead of thinned frames (1.0 = all).
_C.DATASET.TRAIN_RECORDING_FRACTION = 1.0
# Which draw the two cuts above take, so a small cut can be repeated over seeds.
_C.DATASET.TRAIN_SUBSET_SEED = 0
# Distribution-match ablation: 10 per-0.1-mu-bin keep probs for train path cells, None = off
_C.DATASET.MU_MATCH_KEEP = None
_C.DATASET.OVERSAMPLE_MU_MAX = 0.6
_C.DATASET.OVERSAMPLE_OBSERVABILITY_THRESH = 0.6
# LDS exponent for the mu loss weight: w = (1/density)^alpha. Higher = more low-mu loss mass.
_C.DATASET.LDS_ALPHA = 0.75
# Extra low-mu states per train window harvested from the bag (same rule as oversampling), 0 = off
_C.DATASET.HARVEST_LOWMU_EXTRA = 0
# Only harvest within this many seconds of the window time (0 = unlimited)
_C.DATASET.HARVEST_TIME_WINDOW = 0.0
# Zero the loss weight of cells not visible in any input frustum
_C.DATASET.VISIBILITY_GATE = False
# Also require the input depth not to occlude the state; only with VISIBILITY_GATE
_C.DATASET.OCCLUSION_GATE = False

_C.AUGMENTATIONS = EasyDict()
_C.AUGMENTATIONS.HORIZ_FLIP = 0.5
_C.AUGMENTATIONS.PCLOUD_DROPOUT = 0.3
_C.AUGMENTATIONS.MAX_TRANSLATION = 2.0
_C.AUGMENTATIONS.MAX_ROTATION = 1.5708
# ColorJitter (brightness, contrast, saturation, hue)
_C.AUGMENTATIONS.COLOR_JITTER = [0.2, 0.2, 0.2, 0.1]
# Probability of Gaussian blur per frame
_C.AUGMENTATIONS.BLUR_PROB = 0.0

_C.SEED = 42


def merge_cfgs(base_cfg, new_cfg):
    config = copy.deepcopy(base_cfg)
    for key, val in new_cfg.items():
        if key in config:
            if type(config[key]) is EasyDict:
                config[key] = merge_cfgs(config[key], val)
            else:
                config[key] = val
        else:
            sys.exit("key {} doesn't exist in the default configs".format(key))
    return config


def apply_tier_ranges(cfg):
    """TRAIN_/VALID_TIME_RANGES from tiers.json; exits if the mission lists disagree with it."""
    from shared_utils import tiers

    for key, tier, missions in (
            ("TRAIN_TIME_RANGES", tiers.ID_TRAIN, cfg.DATASET.TRAIN_MISSIONS),
            ("VALID_TIME_RANGES", "id_heldout", cfg.DATASET.VALID_MISSIONS)):
        expected = set(tiers.tier_missions(cfg.DATASET.TRAIN_MISSIONS).get(tier, []))
        if expected != set(missions):
            sys.exit(f"DATASET.{tier.upper()} missions {sorted(missions)} != tiers.json "
                     f"{sorted(expected)}; fix the config or the tiering")
        ranges = {m: r for m, r in tiers.time_ranges(tier).items() if m in set(missions)}
        cfg.DATASET[key] = ranges or None
    return cfg


def get_cfg(cfg_file):
    cfg = copy.deepcopy(_C)
    if cfg_file is None:
        return cfg
    with open(cfg_file, "r") as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = merge_cfgs(cfg, new_config)
    if cfg.DATASET.TIER_TIME_RANGES:
        cfg = apply_tier_ranges(cfg)
    return cfg
