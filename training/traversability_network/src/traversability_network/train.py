# Adapted from wayfaster repository (https://github.com/matval/wayfaster) wayfaster/train.py
# Modifications:
#   - Dataset -> GrandTourDataset, config loader -> traversability_network.config.
#   - Train/valid sets built from configs.DATASET.TRAIN_MISSIONS / VALID_MISSIONS.
#   - TrainingModule (trainer.py) is reused unchanged.
import argparse

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from .config import get_cfg
from .data.grandtour_dataset import GrandTourDataset
from .network.trainer import TrainingModule


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="path to config yaml"
    )
    args = parser.parse_args()
    return get_cfg(args.cfg_file)


def main():
    torch.set_float32_matmul_precision("high")
    # Fixed input shapes (drop_last) -> let cudnn autotune the conv algorithms.
    torch.backends.cudnn.benchmark = True
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    configs = parse_config()
    print("configs:\n", configs)

    pl.seed_everything(configs.SEED, workers=True)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

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
        pretrained_dict = torch.load(configs.MODEL.LOAD_NETWORK, map_location="cpu")[
            "state_dict"
        ]
        model.load_state_dict(pretrained_dict)

    wandb_logger = WandbLogger(project="master-thesis-lmertens", log_model=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="checkpoint_{epoch}-{step}",
        monitor="valid/loss",
        save_weights_only=True,
        save_last=True,
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        devices=configs.TRAINING.DEVICES,
        num_nodes=1,
        gradient_clip_val=10,
        sync_batchnorm=True,
        enable_model_summary=True,
        accelerator="gpu",
        logger=wandb_logger,
        default_root_dir="checkpoints",
        max_epochs=configs.TRAINING.EPOCHS,
        check_val_every_n_epoch=1,
        precision=configs.TRAINING.PRECISION,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback, lr_monitor, RichProgressBar(leave=True)],
    )

    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
