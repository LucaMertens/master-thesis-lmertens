"""Overfit smoke test: train on ETH-1 and check traversability error drops on it.

Exercises the full pipeline (GrandTourDataset -> TravNet -> Lightning loop) on a
small ETH-1 subset and asserts the model fits it. Skips when the dataset or a
GPU is unavailable. Run directly for a verbose report incl. the convergence
curve:

    uv run --project training/traversability_network python \
        training/traversability_network/tests/test_smoke_train_eth1.py
"""

import os

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from traversability_network.config import get_cfg
from traversability_network.data.grandtour_dataset import GrandTourDataset
from traversability_network.network.trainer import TrainingModule

MISSION = "ETH-1"
SUBSET = 16
EPOCHS = 100
CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "grandtour.yaml")


class _SilentModule(TrainingModule):
    """TrainingModule without wandb logging and with a flat LR.

    The default CosineAnnealingLR(T_max=10) decays the LR to its minimum
    mid-run, which distorts a short overfit. A flat optimizer isolates whether
    the pipeline can fit the data.
    """

    def visualize(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class _Curve(pl.Callback):
    """Records the per-epoch train traversability loss."""

    def __init__(self):
        self.history = []

    def on_train_epoch_end(self, trainer, pl_module):
        value = trainer.callback_metrics.get("train_trav_loss")
        if value is not None:
            self.history.append(float(value))


def _smoke_cfg():
    cfg = get_cfg(CFG)
    cfg.TRAINING.BATCHSIZE = 4
    cfg.TRAINING.WORKERS = 63
    cfg.TRAINING.PRECISION = 32
    cfg.TRAINING.EPOCHS = EPOCHS
    cfg.AUGMENTATIONS.HORIZ_FLIP = 0.0
    cfg.AUGMENTATIONS.PCLOUD_DROPOUT = 0.0
    cfg.AUGMENTATIONS.MAX_TRANSLATION = 0.0
    cfg.AUGMENTATIONS.MAX_ROTATION = 0.0
    return cfg


@torch.no_grad()
def _mean_trav_error(model, loader, device):
    model.to(device)
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = [x.to(device) for x in batch]
        color, pcloud, K, ext, path, target, weights, depth_t, _ = batch
        trav_map, _, _ = model.model(color, pcloud, K, ext, depth_t)
        _, error = model.trav_criterion(path, trav_map, target, weights)
        total += float(error)
        n += 1
    return total / n


def _build():
    cfg = _smoke_cfg()
    if not os.path.isdir(os.path.join(cfg.DATASET.DATA_ROOT, MISSION)):
        pytest.skip(f"{MISSION} not available under {cfg.DATASET.DATA_ROOT}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    full = GrandTourDataset(cfg, [MISSION], train=True)
    subset = Subset(full, list(range(min(SUBSET, len(full)))))
    loader = DataLoader(
        subset,
        batch_size=cfg.TRAINING.BATCHSIZE,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.TRAINING.WORKERS,
        pin_memory=True,
    )
    eval_loader = DataLoader(subset, batch_size=cfg.TRAINING.BATCHSIZE, shuffle=False)
    return cfg, loader, eval_loader


def _run(progress_bar=True):
    pl.seed_everything(0, workers=True)
    cfg, loader, eval_loader = _build()
    device = torch.device("cuda")
    model = _SilentModule(cfg).to(device)

    before = _mean_trav_error(model, eval_loader, device)
    model.train()

    curve = _Curve()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        max_epochs=cfg.TRAINING.EPOCHS,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=progress_bar,
        num_sanity_val_steps=0,
        callbacks=[curve],
    )
    trainer.fit(model, loader)

    after = _mean_trav_error(model, eval_loader, device)
    return before, after, curve.history


def test_overfits_eth1():
    before, after, _ = _run()
    print(f"\ntrav_error  before={before:.4f}  after={after:.4f}")
    assert torch.isfinite(torch.tensor(after))
    assert after < before * 0.7, (
        f"model did not fit ETH-1 (before {before:.4f}, after {after:.4f})"
    )


if __name__ == "__main__":
    before, after, history = _run(progress_bar=True)
    print(
        f"\nETH-1 overfit smoke test: trav_error before={before:.4f} after={after:.4f}"
    )
    print("train_trav_loss per 10 epochs:")
    for i in range(0, len(history), 10):
        print(f"  epoch {i:3d}: {history[i]:.4f}")
    if history:
        print(f"  epoch {len(history) - 1:3d}: {history[-1]:.4f}  (final)")
