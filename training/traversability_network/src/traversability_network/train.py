# Adapted from wayfaster repository (https://github.com/matval/wayfaster) wayfaster/train.py
# Modifications:
#   - Dataset -> GrandTourDataset, config loader -> traversability_network.config.
#   - Train/valid sets built from configs.DATASET.TRAIN_MISSIONS / VALID_MISSIONS.
#   - TrainingModule (trainer.py) is reused unchanged.
import argparse
import os
import tempfile

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from .config import get_cfg
from .data.grandtour_dataset import GrandTourDataset
from .network.trainer import TrainingModule

# Big-storage volume; home NFS is small and shared, it fills up mid-run.
CKPT_BASE = "/datahome/lmertens/checkpoints"


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="path to config yaml"
    )
    args = parser.parse_args()
    return get_cfg(args.cfg_file)


def main():
    # Checkpoint atomic-save stages to a tempfile then renames into CKPT_BASE.
    # Default /tmp is a different device, so the rename crosses devices and fails
    # (Errno 18). Keep the temp on the same fs as the checkpoints.
    ckpt_tmp = os.path.join(CKPT_BASE, ".tmp")
    os.makedirs(ckpt_tmp, exist_ok=True)
    os.environ["TMPDIR"] = ckpt_tmp
    tempfile.tempdir = ckpt_tmp

    torch.set_float32_matmul_precision("high")
    # Fixed input shapes (drop_last) -> let cudnn autotune the conv algorithms.
    torch.backends.cudnn.benchmark = True
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    configs = parse_config()
    print("configs:\n", configs)

    pl.seed_everything(configs.SEED, workers=True)

    brightness, contrast, saturation, hue = configs.AUGMENTATIONS.COLOR_JITTER
    # Tensor-mode transforms applied to the stacked (T, C, H, W) window in the
    # dataloader: parameters are drawn once per window, not per frame.
    steps = [
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
    ]
    if configs.AUGMENTATIONS.BLUR_PROB > 0:
        steps.append(transforms.RandomApply(
            [transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=configs.AUGMENTATIONS.BLUR_PROB
        ))
    transform = transforms.Compose(steps)

    train_dataset = GrandTourDataset(
        configs, configs.DATASET.TRAIN_MISSIONS, transform=transform, train=True
    )
    valid_dataset = GrandTourDataset(
        configs, configs.DATASET.VALID_MISSIONS, weights=train_dataset.weights
    )

    # forkserver workers are forked from a single-threaded server started before
    # any CUDA use, so they sidestep the fork-after-CUDA deadlock that plain fork
    # hits once sanity validation has initialised CUDA. persistent_workers keeps
    # them alive across epochs so the cold start (re-import + dataset unpickle) is
    # paid once, not every epoch.
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

    model = TrainingModule(configs)

    if configs.MODEL.LOAD_NETWORK is not None:
        print("Loading saved network from {}".format(configs.MODEL.LOAD_NETWORK))
        # weights_only=False: our own checkpoints carry EasyDict hparams.
        # strict=False: older checkpoints lack newer metric buffers (e.g. val_*_vis).
        pretrained_dict = torch.load(
            configs.MODEL.LOAD_NETWORK, map_location="cpu", weights_only=False
        )["state_dict"]
        # Normalize torch.compile's "._orig_mod" prefix to match this run's keys
        # (COMPILE can differ between the saved run and this one): strip it, then
        # re-add it if this run's own model is the one that's compiled.
        pretrained_dict = {k.replace("._orig_mod", ""): v for k, v in pretrained_dict.items()}
        own = model.state_dict()
        if any("._orig_mod" in k for k in own):
            pretrained_dict = {
                k.replace("model.", "model._orig_mod.", 1) if k.startswith("model.") else k: v
                for k, v in pretrained_dict.items()
            }
        # Warm start across head changes (e.g. plain -> evidential/binned mu head):
        # drop shape-mismatched keys so the new head keeps its fresh init, since
        # load_state_dict raises on those regardless of strict=False.
        mismatched = [k for k, v in pretrained_dict.items() if k in own and own[k].shape != v.shape]
        for k in mismatched:
            del pretrained_dict[k]
        if mismatched:
            print(f"  dropped {len(mismatched)} shape-mismatched keys: {mismatched}")
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
        if missing or unexpected:
            print(f"  non-strict load: missing={missing} unexpected={unexpected}")

    wandb_logger = WandbLogger(
        project="master-thesis-lmertens", log_model=False, save_code=True
    )

    # Persist the exact uncommitted diff so a run is reproducible even without a
    # commit; wandb records the commit hash but its auto diff.patch comes back empty.
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    if rank_zero_only.rank == 0:
        import subprocess

        try:
            root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], text=True
            ).strip()
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
        save_top_k=-1,  # Save every nth checkpoint, do not check for best.
        every_n_epochs=5,  # Save every 5 epochs
    )
    # valid/trav_loss stays flat straight through mean collapse (dominated by the
    # common high-mu mass), so it can't see the failure it's meant to catch.
    # valid/balanced_mae macro-averages MAE per true-mu bin (equal weight per bin),
    # which degrades monotonically as low-mu predictions collapse toward the mean.
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
        min_epochs=20,  # early stopping (patience 10) can't fire before this many epochs
        check_val_every_n_epoch=1,
        precision=configs.TRAINING.PRECISION,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback, best_callback, early_stopping, lr_monitor, RichProgressBar(leave=True)],
    )

    # weights_only=False: our own checkpoints carry EasyDict hparams (see LOAD_NETWORK above).
    trainer.fit(
        model, train_loader, valid_loader,
        ckpt_path=configs.MODEL.RESUME_CHECKPOINT, weights_only=False,
    )


if __name__ == "__main__":
    main()
