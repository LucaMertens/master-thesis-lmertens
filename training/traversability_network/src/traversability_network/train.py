# Training entrypoint, adapted from wayfaster's train.py (https://github.com/matval/wayfaster).
import argparse
import os
import tempfile

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from . import adaptations
from .adaptations.grandtour_dataset import GrandTourDataset
from .config import get_cfg
from .network.trainer import TrainingModule

CKPT_BASE = "/datahome/lmertens/checkpoints"
# wandb appends its own "wandb/" subdir, so runs land in WANDB_DIR/wandb/run-...
WANDB_DIR = "/datahome/lmertens"


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="path to config yaml"
    )
    # ADAPTATION: safe in every launch command, starts fresh if the TAG has no run yet
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="WANDB_ID",
        help="continue this TAG's last run, or the given wandb run id",
    )
    args = parser.parse_args()
    return get_cfg(args.cfg_file), args.resume


def main():
    # Atomic checkpoint save renames a tempfile into CKPT_BASE; /tmp is another device (Errno 18)
    ckpt_tmp = os.path.join(CKPT_BASE, ".tmp")
    os.makedirs(ckpt_tmp, exist_ok=True)
    os.environ["TMPDIR"] = ckpt_tmp
    tempfile.tempdir = ckpt_tmp

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    configs, resume = parse_config()
    print("configs:\n", configs)

    pl.seed_everything(configs.SEED, workers=True)

    brightness, contrast, saturation, hue = configs.AUGMENTATIONS.COLOR_JITTER
    # Applied to the stacked (T, C, H, W) window: one draw per window, not per frame
    steps = [
        transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        ),
    ]
    if configs.AUGMENTATIONS.BLUR_PROB > 0:
        steps.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.5))],
                p=configs.AUGMENTATIONS.BLUR_PROB,
            )
        )
    transform = transforms.Compose(steps)
    if getattr(configs.MODEL, "BACKBONE", "resnet34") == "dino":
        # Features are precomputed, photometric augs would not reach them
        transform = None

    if getattr(configs.DATASET, "NAME", "grandtour") == "wayfaster":
        from .network.dataloader import Dataset as WayfasterDataset

        train_dataset = WayfasterDataset(
            configs, configs.DATASET.TRAIN_DIRS, transform=transform, train=True
        )
        valid_dataset = WayfasterDataset(
            configs, configs.DATASET.VALID_DIRS, weights=train_dataset.weights
        )
    else:
        train_dataset = GrandTourDataset(
            configs, configs.DATASET.TRAIN_MISSIONS, transform=transform, train=True
        )
        valid_dataset = GrandTourDataset(
            configs, configs.DATASET.VALID_MISSIONS, weights=train_dataset.weights
        )

    # ADAPTATION: human W-RIZZ pairs (adaptations/human_pairs.py). The sampler keeps both
    # windows of a pair in one batch; None without HUMAN_WEIGHT.
    human_sampler = adaptations.attach_human_pairs(configs, train_dataset, valid_dataset)

    # forkserver avoids the fork-after-CUDA deadlock; persistent workers skip the cold start
    workers = configs.TRAINING.WORKERS
    loader_kwargs = dict(
        batch_size=configs.TRAINING.BATCHSIZE,
        num_workers=workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=workers > 0,
        multiprocessing_context="forkserver" if workers > 0 else None,
    )
    if human_sampler is None:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    else:
        for key in ("batch_size", "drop_last"):
            loader_kwargs.pop(key)
        train_loader = DataLoader(train_dataset, batch_sampler=human_sampler, **loader_kwargs)
        loader_kwargs.update(batch_size=configs.TRAINING.BATCHSIZE, drop_last=True)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)

    model = TrainingModule(configs)

    if configs.MODEL.LOAD_NETWORK is not None:
        print("Loading saved network from {}".format(configs.MODEL.LOAD_NETWORK))
        # ADAPTATION: see adaptations/checkpoint.py
        adaptations.load_pretrained(model, configs.MODEL.LOAD_NETWORK)

    # ADAPTATION: WANDB_RESUME_ID appends a continued run to its original history
    resume_id = getattr(configs.TRAINING, "WANDB_RESUME_ID", None)
    resume_ckpt = configs.MODEL.RESUME_CHECKPOINT
    # ADAPTATION: --resume looks up wandb id and last.ckpt from the TAG; config keys win.
    # Must not use wandb_logger.experiment.id (rank-0 only), all ranks need the same ckpt_path.
    if resume is not None:
        found_id, found_ckpt = adaptations.lookup_run(
            CKPT_BASE, configs.TAG, None if resume is True else resume
        )
        resume_id = resume_id or found_id
        resume_ckpt = resume_ckpt or found_ckpt
        print(f"resume: wandb id {resume_id}, checkpoint {resume_ckpt}")

    wandb_logger = WandbLogger(
        project="master-thesis-lmertens", name=configs.TAG,
        save_dir=WANDB_DIR,
        log_model=False, save_code=True,
        **({"id": resume_id, "resume": "must"} if resume_id else {}),
        notes=configs.DESCRIPTION or None,
    )

    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    # ADAPTATION: TAG -> wandb id, for --resume
    if rank_zero_only.rank == 0:
        adaptations.record_run(CKPT_BASE, configs.TAG, wandb_logger.experiment.id)

    # ADAPTATION: log per-split mu statistics to wandb
    if rank_zero_only.rank == 0:
        for name, ds in (("train", train_dataset), ("valid", valid_dataset)):
            stats = adaptations.dataset_stats(ds)
            print(adaptations.format_stats(name, stats))
            adaptations.log_stats(wandb_logger, name, stats)

    # Save the uncommitted diff; wandb's own diff.patch comes back empty
    if rank_zero_only.rank == 0:
        import subprocess

        try:
            root = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], text=True
            ).strip()
            diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=root, text=True)
            patch = os.path.join(
                ckpt_tmp, f"uncommitted_{wandb_logger.experiment.id}.patch"
            )
            with open(patch, "w") as fh:
                fh.write(diff)
            wandb_logger.experiment.save(patch, base_path=ckpt_tmp, policy="now")
        except Exception as e:
            print(f"git diff capture failed: {e}")

    run_dir = f"{CKPT_BASE}/{wandb_logger.experiment.id}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="checkpoint_{epoch}-{step}",
        save_top_k=-1,
        every_n_epochs=configs.TRAINING.CHECKPOINT_EVERY_N_EPOCHS,
        # ADAPTATION: epoch end only, else sub-epoch validation writes duplicate checkpoints
        save_on_train_epoch_end=True,
    )
    # ADAPTATION: rolling last.ckpt for --resume (adaptations/resume.py)
    resume_callback = adaptations.ResumeCheckpoint(run_dir)
    # ADAPTATION: select on CKPT_MONITOR, not valid/loss (dominated by high-mu cells)
    best_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_{epoch}-{step}",
        monitor=configs.TRAINING.CKPT_MONITOR,
        mode=configs.TRAINING.CKPT_MODE,
        save_top_k=1,
        save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # ADAPTATION: archive checkpoints on a validation schedule (adaptations/archive.py)
    every_n_val = configs.TRAINING.CHECKPOINT_EVERY_N_VALIDATIONS
    val_callbacks = []
    if every_n_val:
        val_callbacks.append(adaptations.ArchiveCheckpoints(
            run_dir, configs.TRAINING.CHECKPOINT_DENSE_EPOCHS, every_n_val
        ))

    trainer = pl.Trainer(
        devices=configs.TRAINING.DEVICES,
        num_nodes=1,
        gradient_clip_val=10,
        # ADAPTATION: larger effective batch without more VRAM
        accumulate_grad_batches=configs.TRAINING.ACCUMULATE_GRAD_BATCHES,
        sync_batchnorm=True,
        enable_model_summary=True,
        accelerator="gpu",
        logger=wandb_logger,
        default_root_dir=CKPT_BASE,
        max_epochs=configs.TRAINING.EPOCHS,
        min_epochs=1,
        check_val_every_n_epoch=1,
        # ADAPTATION: sub-epoch validation, best low-mu transfer is reached within epoch 1
        val_check_interval=configs.TRAINING.VAL_CHECK_INTERVAL,
        precision=configs.TRAINING.PRECISION,
        strategy=DDPStrategy(find_unused_parameters=False),
        # ADAPTATION: the human batch sampler cannot be wrapped in a DistributedSampler
        use_distributed_sampler=human_sampler is None,
        callbacks=[
            checkpoint_callback,
            resume_callback,
            best_callback,
            lr_monitor,
            RichProgressBar(leave=True),
        ] + val_callbacks,
    )

    # weights_only=False: our checkpoints carry EasyDict hparams
    trainer.fit(
        model,
        train_loader,
        valid_loader,
        ckpt_path=resume_ckpt,
        weights_only=False,
    )


if __name__ == "__main__":
    main()
