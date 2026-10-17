"""Overfit smoke test: small ETH-1 subset, traversability error must drop."""

# ruff: noqa: E402

import os
import subprocess

def _get_free_gpu():
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        memory_used = [int(x) for x in smi_out.strip().split("\n")]
        return memory_used.index(min(memory_used))
    except Exception:
        return 0

os.environ["CUDA_VISIBLE_DEVICES"] = str(_get_free_gpu())

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from traversability_network.config import get_cfg
from traversability_network.adaptations.grandtour_dataset import GrandTourDataset
from traversability_network.network.trainer import TrainingModule

MISSION = "ETH-1"
SUBSET = 16
EPOCHS = 100
CFG = os.path.join(os.path.dirname(__file__), "..", "configs", "grandtour.yaml")


class _SilentModule(TrainingModule):
    """TrainingModule without wandb logging and with a flat LR (cosine decay distorts short overfits)."""

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
        value = trainer.callback_metrics.get("train/trav_loss")
        if value is not None:
            self.history.append(float(value))


def _smoke_cfg():
    cfg = get_cfg(CFG)
    cfg.TRAINING.BATCHSIZE = 1
    cfg.TRAINING.WORKERS = 63
    cfg.TRAINING.PRECISION = 32
    cfg.TRAINING.EPOCHS = EPOCHS
    cfg.AUGMENTATIONS.HORIZ_FLIP = 0.0
    cfg.AUGMENTATIONS.PCLOUD_DROPOUT = 0.0
    cfg.AUGMENTATIONS.MAX_TRANSLATION = 0.0
    cfg.AUGMENTATIONS.MAX_ROTATION = 0.0
    cfg.DATASET.TRAIN_MISSIONS = [MISSION]
    cfg.DATASET.VALID_MISSIONS = []
    return cfg


@torch.no_grad()
def _mean_trav_error(model, loader, device):
    model.to(device)
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = [x.to(device) for x in batch]
        color, pcloud, K_full, K, ext, path, target, weights, depth_t, depth_m, *_extra = batch
        trav_map, _, _, _ = model.model(color, pcloud, K_full, K, ext, depth_t)
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
    model.prepare_data()

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
