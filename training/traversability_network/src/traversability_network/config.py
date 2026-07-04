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
# Balanced MSE (BMC, Ren et al. 2022) on the mu term: initial noise sigma of the
# learnable scale; 0 = off (plain smooth-L1). Changes the trav_loss scale.
_C.TRAINING.BMC_SIGMA_INIT = 0.0
_C.TRAINING.VIS_INTERVAL = 500
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
# Mu head: 0 = sigmoid regression (WayFASTER default); K > 0 = categorical
# distribution over K mu bins trained with soft two-hot CE, expectation readout.
_C.MODEL.MU_BINS = 0
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
# Directory holding one sub-directory per mission (images, depth, odom, intrinsics, ...).
_C.DATASET.DATA_ROOT = str(EXTRACTED_ROOT)
# Directory holding one sub-directory per mission with rhe/ and simple/ label outputs.
_C.DATASET.LABELS_ROOT = str(LABELS_ROOT)
# Local-SSD root with pre-resized (INPUT_SIZE) color images; produced by
# preprocessing/grandtour_extraction's resize_images.py (see its --width/--height).
# None = read native-res images from DATA_ROOT (slow over NFS).
_C.DATASET.RESIZED_ROOT = str(DATALOCAL_ROOT / "grand_tour_dataset_resized")
_C.DATASET.DEPTH_CACHE_ROOT = str(DATALOCAL_ROOT / "grand_tour_dataset_depth")
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
# Zero the loss weight of horizon cells not visible in any input frame's camera
# frustum. Invisible cells are unpredictable from the input; supervising them only
# pulls predictions toward the label mean (~45% of low-mu cells are invisible).
_C.DATASET.VISIBILITY_GATE = False

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
