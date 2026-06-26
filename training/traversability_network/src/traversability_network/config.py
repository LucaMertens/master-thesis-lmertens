# Taken from wayfaster repository (https://github.com/matval/wayfaster) train/train_configs.py
# Modifications:
#   - DATASET section reworked for the GrandTour layout: DATA_ROOT + named
#     TRAIN_MISSIONS / VALID_MISSIONS lists instead of CSV-based bag folders.
#   - Added DATASET.TIME_TOLERANCE and DATASET.MAX_TIME_GAP for timestamp matching.
#   - Added MODEL.LABEL_COLUMNS to pick (mu, nu) out of rhe_trav_labels.npy.
import copy
import sys

import yaml
from config.paths import EXTRACTED_ROOT, LABELS_ROOT
from easydict import EasyDict

_C = EasyDict()

_C.TAG = "grandtour"

_C.TRAINING = EasyDict()
_C.TRAINING.EPOCHS = 50
_C.TRAINING.BATCHSIZE = 4
_C.TRAINING.WORKERS = 8
# Ampere bf16: fp16 speed/memory, fp32 range (no overflow / loss scaling)
_C.TRAINING.PRECISION = "bf16-mixed"
_C.TRAINING.DT = 0.1  # time step in seconds
_C.TRAINING.HORIZON = 300  # horizon in number of points
_C.TRAINING.GAMMA = 0.998
_C.TRAINING.DEPTH_WEIGHT = 0.1
_C.TRAINING.VIS_INTERVAL = 500
_C.TRAINING.VERBOSE = True

# model parameters
_C.MODEL = EasyDict()
_C.MODEL.LOAD_NETWORK = None
_C.MODEL.DOWNSAMPLE = 8
_C.MODEL.LATENT_DIM = 64
_C.MODEL.TIME_LENGTH = 6
_C.MODEL.PREDICT_DEPTH = True
_C.MODEL.TRAIN_DEPTH = True
_C.MODEL.FUSE_PCLOUD = True
_C.MODEL.INPUT_SIZE = [480, 360]  # (W, H); Alphasense native is 1440x1080 (4:3)
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
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4

_C.DATASET = EasyDict()
# Directory holding one sub-directory per mission (images, depth, odom, intrinsics, ...).
_C.DATASET.DATA_ROOT = str(EXTRACTED_ROOT)
# Directory holding one sub-directory per mission with rhe/ and simple/ label outputs.
_C.DATASET.LABELS_ROOT = str(LABELS_ROOT)
_C.DATASET.TRAIN_MISSIONS = []
_C.DATASET.VALID_MISSIONS = []
_C.DATASET.TIME_TOLERANCE = 0.05  # s, max camera<->RHE/depth timestamp mismatch
_C.DATASET.MAX_TIME_GAP = 0.5  # s, skip image sequences crossing a larger gap
# Low-mu oversampling: replicate loss horizons with any cell (mu <= MAX & observability >= MIN)
# until they reach OVERSAMPLE_FRACTION of the (train) index list.
_C.DATASET.OVERSAMPLE_FRACTION = 0.40
_C.DATASET.OVERSAMPLE_MU_MAX = 0.6
_C.DATASET.OVERSAMPLE_OBSERVABILITY_THRESH = 0.6
# LDS exponent for the mu loss weight: w = (1/density)^alpha. Higher = more low-mu loss mass.
_C.DATASET.LDS_ALPHA = 0.75

_C.AUGMENTATIONS = EasyDict()
_C.AUGMENTATIONS.HORIZ_FLIP = 0.5
_C.AUGMENTATIONS.PCLOUD_DROPOUT = 0.3
_C.AUGMENTATIONS.MAX_TRANSLATION = 2.0
_C.AUGMENTATIONS.MAX_ROTATION = 1.5708

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
