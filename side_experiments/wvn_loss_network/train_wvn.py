"""Training entry point for the WVN loss experiment."""

import argparse
import os
import sys
import tempfile

import pytorch_lightning as pl
import torch
from easydict import EasyDict
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traversability_network.config as tn_config
from traversability_network.adaptations.grandtour_dataset import GrandTourDataset
from wvn_module import WvnTrainingModule

CKPT_BASE = "/datahome/lmertens/checkpoints"

# WVN section, registered so merge_cfgs accepts it
tn_config._C.WVN = EasyDict()
tn_config._C.WVN.GATE_PATH = "side_experiments/wvn_loss_network/output/gate.npz"
tn_config._C.WVN.TAU_QUANTILE = 0.95  # per-window tau quantile
tn_config._C.WVN.MIN_CELLS = 20      # min cells to calibrate tau
tn_config._C.WVN.WEIGHT = 0.05       # loss weight
tn_config._C.WVN.VIS_MIN = 0.5       # min splat mass to count as visible
tn_config._C.WVN.MAX_DIST = 6.0      # m, gate was fit on 1-6 m views
tn_config._C.WVN.Z_MAX = 0.5         # m, ignore z-levels above (canopy)
tn_config._C.WVN.LIGHT_BEV = False   # local-RF BEV stack, see wvn_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="side_experiments/wvn_loss_network/wvn_dino.yaml")
    args = parser.parse_args()
    configs = tn_config.get_cfg(args.cfg_file)
    print("configs:\n", configs)

    ckpt_tmp = os.path.join(CKPT_BASE, ".tmp")
    os.makedirs(ckpt_tmp, exist_ok=True)
    os.environ["TMPDIR"] = ckpt_tmp
    tempfile.tempdir = ckpt_tmp

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    pl.seed_everything(configs.SEED, workers=True)

    assert configs.MODEL.BACKBONE == "dino", "the WVN gate needs the cached DINO features"
    train_dataset = GrandTourDataset(configs, configs.DATASET.TRAIN_MISSIONS, train=True)
    valid_dataset = GrandTourDataset(configs, configs.DATASET.VALID_MISSIONS, weights=train_dataset.weights)

    workers = configs.TRAINING.WORKERS
    loader_kwargs = dict(
        batch_size=configs.TRAINING.BATCHSIZE,
        num_workers=workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=workers > 0,
        multiprocessing_context="forkserver" if workers > 0 else None,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)

    model = WvnTrainingModule(configs)

    wandb_logger = WandbLogger(project="master-thesis-lmertens", log_model=False, save_code=True)

    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if rank_zero_only.rank == 0:
        import subprocess

        try:
            root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
            diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root, text=True)
            patch = os.path.join(ckpt_tmp, f"uncommitted_{wandb_logger.experiment.id}.patch")
            with open(patch, "w") as fh:
                fh.write(diff)
            wandb_logger.experiment.save(patch, base_path=ckpt_tmp, policy="now")
        except Exception as e:
            print(f"git diff capture failed: {e}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{CKPT_BASE}/{wandb_logger.experiment.id}",
        filename="checkpoint_{epoch}-{step}",
        monitor="valid/loss",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=5,
    )
    best_callback = ModelCheckpoint(
        dirpath=f"{CKPT_BASE}/{wandb_logger.experiment.id}",
        filename="best_{epoch}-{step}",
        monitor="valid/balanced_mae",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(monitor="valid/balanced_mae", mode="min", patience=10)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        devices=configs.TRAINING.DEVICES,
        num_nodes=1,
        gradient_clip_val=10,
        sync_batchnorm=True,
        enable_model_summary=True,
        accelerator="gpu",
        logger=wandb_logger,
        default_root_dir=CKPT_BASE,
        max_epochs=configs.TRAINING.EPOCHS,
        min_epochs=20,
        check_val_every_n_epoch=1,
        precision=configs.TRAINING.PRECISION,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback, best_callback, early_stopping, lr_monitor, RichProgressBar(leave=True)],
    )

    trainer.fit(model, train_loader, valid_loader,
                ckpt_path=configs.MODEL.RESUME_CHECKPOINT, weights_only=False)


if __name__ == "__main__":
    main()
