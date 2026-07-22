# Taken from wayfaster repository (https://github.com/matval/wayfaster) train/train_configs.py
# Modifications:
#   - DATASET section reworked for the GrandTour layout: DATA_ROOT + named
#     TRAIN_MISSIONS / VALID_MISSIONS lists instead of CSV-based bag folders.
#   - Added DATASET.TIME_TOLERANCE and DATASET.MAX_TIME_GAP for timestamp matching.
#   - Added MODEL.LABEL_COLUMNS to pick (mu, nu) out of rhe_trav_labels.npy.
import copy
import sys

import yaml
from config.paths import DATALOCAL_ROOT, EXTRACTED_ROOT, LABELS_ROOT
from easydict import EasyDict

_C = EasyDict()

_C.TAG = "grandtour"

_C.TRAINING = EasyDict()
_C.TRAINING.EPOCHS = 50
_C.TRAINING.BATCHSIZE = 4
_C.TRAINING.WORKERS = 8
_C.TRAINING.DEVICES = (
    -1
)  # GPUs for the Trainer; -1 = all visible, or an int/list to limit
_C.TRAINING.COMPILE = True  # torch.compile the model; set False to debug in eager
# Ampere bf16: fp16 speed/memory, fp32 range (no overflow / loss scaling)
_C.TRAINING.PRECISION = "bf16-mixed"
_C.TRAINING.DT = 0.1  # time step in seconds
_C.TRAINING.HORIZON = 300  # horizon in number of points
_C.TRAINING.GAMMA = 0.998
_C.TRAINING.DEPTH_WEIGHT = 0.1
# Extra penalty on optimistic errors (predicted trav > label). >1 makes calling
# slippery terrain traversable costlier than the reverse, pushing uncertain
# predictions down instead of onto the high-mu mean.
_C.TRAINING.TRAV_OVERPRED_WEIGHT = 1.0
# smooth-L1 transition point for the mu/nu term. WayFASTER uses 0.1 (L1-like,
# sharp). Larger beta is more quadratic/MSE-like and compresses toward the mean.
_C.TRAINING.SMOOTH_L1_BETA = 0.1
# trav_loss normalization (adaptations.reduce_trav_loss): "weighted" (adapted)
# divides by weight mass so the loss scale is batch-independent; "mean" (original
# WayFASTER) is the plain mean of error*weight over the masked cells.
_C.TRAINING.LOSS_NORM = "weighted"
# Mu loss selector. None = infer from the legacy MU_BINS/BMC_SIGMA_INIT/EVIDENTIAL
# flags (back-compat). When set it is authoritative: one of "regression"
# (smooth-L1), "bmc" (needs BMC_SIGMA_INIT > 0), "binned_ce" (needs MU_BINS > 0),
# "class3" (needs MU_BINS == number of classes), or "evidential".
_C.TRAINING.LOSS_TYPE = None
# 3-class mu head ("class3"): hard classes from thresholding label mu, trained with
# logit-adjusted CE (Menon et al. 2021) - log-priors added to the logits only in the
# training loss, so raw argmax at inference is already imbalance-corrected. Priors
# are the class frequencies over gated train cells (extract_mu_predictions).
_C.TRAINING.MU_CLASS_THRESHOLDS = [0.3, 0.6]
_C.TRAINING.MU_CLASS_PRIORS = [0.0153, 0.026, 0.9587]
# Balanced MSE (BMC, Ren et al. 2022) on the mu term: initial noise sigma of the
# learnable scale; 0 = off (plain smooth-L1). Changes the trav_loss scale.
_C.TRAINING.BMC_SIGMA_INIT = 0.0
# Deep Evidential Regression (Amini et al. 2020) regularizer weight; only used
# when MODEL.EVIDENTIAL is True. Amini et al. use small values (~1e-2 to 1e-1);
# too high collapses evidence everywhere, too low lets it grow unchecked on noise.
_C.TRAINING.EVIDENTIAL_LAMBDA = 0.01
_C.TRAINING.VIS_INTERVAL = 500
# Cap on images per visualized panel: batch elements are evenly strided down
# to this many, since consecutive elements are 1-frame-apart sliding windows
# (near-duplicate scenes) rather than independent samples.
_C.TRAINING.VIS_BATCH_SIZE = 8
# Extra wandb debug panels (see TrainingModule.visualize), one flag per stage:
# label supervision on the camera/BEV, per-frame path reprojection (ego-motion
# and extrinsics consistency), input depth blended over RGB (alignment).
_C.TRAINING.DEBUG_VIS_LABELS = False
_C.TRAINING.DEBUG_VIS_GEOM = False
_C.TRAINING.DEBUG_VIS_DEPTH = False
_C.TRAINING.VERBOSE = True

# model parameters
_C.MODEL = EasyDict()
_C.MODEL.LOAD_NETWORK = None
_C.MODEL.RESUME_CHECKPOINT = None  # full trainer state (weights+optimizer+epoch), for resuming an interrupted run
_C.MODEL.DOWNSAMPLE = 8
_C.MODEL.LATENT_DIM = 64
_C.MODEL.TIME_LENGTH = 6
_C.MODEL.PREDICT_DEPTH = True
_C.MODEL.TRAIN_DEPTH = True
_C.MODEL.FUSE_PCLOUD = True
# Nearest-subsample factor for the GPU occupancy voxelization (1 = full-res depth).
# Higher = cheaper voxelize, but thins far-field occupancy (see check_voxel_downscale.py).
_C.MODEL.VOXEL_DEPTH_DOWNSAMPLE = 1
_C.MODEL.INPUT_SIZE = [480, 360]  # (W, H); Alphasense native is 1440x1080 (4:3)
# Freeze the ImageNet-pretrained encoder stages (requires_grad=False, BN held in
# eval): only the fresh neck + heads train. Stronger than BACKBONE_LR_FACTOR.
_C.MODEL.FREEZE_BACKBONE = False
# Image backbone: "resnet34" (trainable, WayFASTER default) or "dino" (frozen
# patch features precomputed by grandtour_extraction's cache_dino_features.py,
# read from DATASET.DINO_CACHE_ROOT; only a small neck trains).
# NB: DINOv3 features must be L2-normalized, not per-dim standardized (drops AUROC
# to chance) - the DinoNeck does this. See side_experiments/dino_mu_probe.
_C.MODEL.BACKBONE = "resnet34"
_C.MODEL.DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
_C.MODEL.DINO_FEAT_DIM = 768
# (W, H) fed to DINO at cache time; must be a patch-size multiple. 480x352 gives
# a 30x22 patch grid (patch 16), upsampled to the stride-8 45x60 grid by the neck.
_C.MODEL.DINO_INPUT_SIZE = [480, 352]
_C.MODEL.DINO_NECK_DIM = 256
# Mu head: 0 = sigmoid regression (WayFASTER default); K > 0 = categorical
# distribution over K mu bins trained with soft two-hot CE, expectation readout.
_C.MODEL.MU_BINS = 0
# Probe-as-layer mu path: linear per-pixel MU_BINS-class readout on the L2-normed
# frozen DINO features, lifted to BEV geometrically (detached depth weights,
# mass-normalized mean over frames/z); no learnable mu decoder, BEV head emits nu
# only. Tests whether the linear image-space signal survives the geometry with
# no network capacity in the mu path.
_C.MODEL.MU_PIXEL_LINEAR = False
# Evidential mu head: predicts a Normal-Inverse-Gamma (gamma, v, alpha, beta)
# instead of a point estimate, trained with the DER NLL + regularizer. Mutually
# exclusive with MU_BINS and BMC_SIGMA_INIT. gamma is the point estimate (still
# in trav_map[:,0]); v, alpha, beta give per-cell epistemic uncertainty for OOD.
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
# LR multiplier for the ImageNet-pretrained encoder stages (1.0 = single LR).
# Lower keeps the pretrained features general instead of memorizing the small dataset.
_C.OPTIMIZER.BACKBONE_LR_FACTOR = 1.0

_C.DATASET = EasyDict()
# "grandtour" (GrandTourDataset, mission layout) or "wayfaster" (the original
# CSV-based WayFASTER dataset via network/dataloader.py, architecture control).
_C.DATASET.NAME = "grandtour"
_C.DATASET.CSV_FILE = "rosbags.csv"
# Data directories for NAME=wayfaster (each holds CSV_FILE + one dir per bag).
_C.DATASET.TRAIN_DIRS = []
_C.DATASET.VALID_DIRS = []
# Directory holding one sub-directory per mission (images, depth, odom, intrinsics, ...).
_C.DATASET.DATA_ROOT = str(EXTRACTED_ROOT)
# Directory holding one sub-directory per mission with rhe/ and simple/ label outputs.
_C.DATASET.LABELS_ROOT = str(LABELS_ROOT)
# Local-SSD root with pre-resized (INPUT_SIZE) color images; produced by
# preprocessing/grandtour_extraction's resize_images.py (see its --width/--height).
# None = read native-res images from DATA_ROOT (slow over NFS).
_C.DATASET.RESIZED_ROOT = str(DATALOCAL_ROOT / "grand_tour_dataset_resized")
_C.DATASET.DEPTH_CACHE_ROOT = str(DATALOCAL_ROOT / "grand_tour_dataset_depth")
# Per-mission frozen DINO patch-feature cache (features.npy + meta.npz), see MODEL.BACKBONE.
_C.DATASET.DINO_CACHE_ROOT = str(DATALOCAL_ROOT / "grand_tour_dataset_dino")
_C.DATASET.TRAIN_MISSIONS = []
_C.DATASET.VALID_MISSIONS = []
# Held-out ID test set and OOD-detection benchmark sets (consumed by eval, not training).
_C.DATASET.TEST_MISSIONS = []
_C.DATASET.NEAR_OOD_MISSIONS = []
_C.DATASET.FAR_OOD_MISSIONS = []
_C.DATASET.TIME_TOLERANCE = 0.05  # s, max camera<->RHE/depth timestamp mismatch
_C.DATASET.MAX_TIME_GAP = 0.5  # s, skip image sequences crossing a larger gap
# Trim leading/trailing stationary frames whose whole label horizon has no motion.
_C.DATASET.TRIM_STATIONARY = True
_C.DATASET.MOTION_MIN_SPEED = 0.01  # m/s, |v| above this counts as moving
_C.DATASET.MOTION_MIN_OMEGA = 0.01  # rad/s, |omega| above this counts as moving
# Low-mu oversampling: replicate loss horizons with any cell (mu <= MAX & observability >= MIN)
# until they reach OVERSAMPLE_FRACTION of the (train) index list.
_C.DATASET.OVERSAMPLE_FRACTION = 0.40
_C.DATASET.OVERSAMPLE_MU_MAX = 0.6
_C.DATASET.OVERSAMPLE_OBSERVABILITY_THRESH = 0.6
# LDS exponent for the mu loss weight: w = (1/density)^alpha. Higher = more low-mu loss mass.
_C.DATASET.LDS_ALPHA = 0.75
# Per train window, append up to this many extra low-mu states (same qualifying rule
# as oversampling) from anywhere in the bag that fall inside the window's grid.
# More views per rare low-mu cell; 0 = off.
_C.DATASET.HARVEST_LOWMU_EXTRA = 0
# Only harvest low-mu states within this many seconds of the window's current time.
# Bounds how far the anchor-frame transform extrapolates the odometry; 0 = unlimited.
_C.DATASET.HARVEST_TIME_WINDOW = 0.0
# Zero the loss weight of horizon cells not visible in any input frame's camera
# frustum. Invisible cells are unpredictable from the input; supervising them only
# pulls predictions toward the label mean (~45% of low-mu cells are invisible).
_C.DATASET.VISIBILITY_GATE = False
# Strengthen the visibility gate with a depth-based occlusion test: a frame only
# counts as seeing a state if the input depth at the projected pixel is not clearly
# in front of it (18% of frustum-gated states are occluded in every viewing frame).
# Only active together with VISIBILITY_GATE.
_C.DATASET.OCCLUSION_GATE = False

_C.AUGMENTATIONS = EasyDict()
_C.AUGMENTATIONS.HORIZ_FLIP = 0.5
_C.AUGMENTATIONS.PCLOUD_DROPOUT = 0.3
_C.AUGMENTATIONS.MAX_TRANSLATION = 2.0
_C.AUGMENTATIONS.MAX_ROTATION = 1.5708
# ColorJitter (brightness, contrast, saturation, hue). Strong photometric jitter can
# erase the subtle appearance cues (wetness, sheen) that correlate with low mu.
_C.AUGMENTATIONS.COLOR_JITTER = [0.2, 0.2, 0.2, 0.1]
# Probability of Gaussian blur per frame (robustness to focus/motion blur).
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


def get_cfg(cfg_file):
    cfg = copy.deepcopy(_C)
    if cfg_file is None:
        return cfg
    with open(cfg_file, "r") as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = merge_cfgs(cfg, new_config)
    return cfg
