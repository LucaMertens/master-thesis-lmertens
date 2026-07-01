"""Validation callbacks for the traversability network."""
import pytorch_lightning as pl
import torch
from torchmetrics import SumMetric


class CalibrationSlope(pl.Callback):
    """Log the slope/intercept of predicted-mu vs true-mu over the whole val set.

    slope ~1 = calibrated, slope ~0 = collapsed to a constant. Uses closed-form
    least squares from DDP-reduced sufficient statistics, so it is one scalar per
    epoch instead of eyeballing the per-bin calibration table.
    """

    def __init__(self, key="val/calib_slope"):
        super().__init__()
        self.key = key
        self.stats = None

    def _reset(self, device):
        self.stats = {k: SumMetric().to(device) for k in ("n", "x", "y", "xx", "xy")}

    def on_validation_epoch_start(self, trainer, pl_module):
        self._reset(pl_module.device)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        x = getattr(pl_module, "_cal_true_mu", None)
        y = getattr(pl_module, "_cal_pred_mu", None)
        if x is None or x.numel() == 0:
            return
        x, y = x.float(), y.float()
        self.stats["n"].update(torch.ones_like(x))
        self.stats["x"].update(x)
        self.stats["y"].update(y)
        self.stats["xx"].update(x * x)
        self.stats["xy"].update(x * y)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.stats is None:
            return
        n = self.stats["n"].compute()
        sx, sy = self.stats["x"].compute(), self.stats["y"].compute()
        sxx, sxy = self.stats["xx"].compute(), self.stats["xy"].compute()
        denom = n * sxx - sx * sx
        if denom <= 0:  # empty val set (e.g. sanity check with no in-bounds cells)
            return
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        # stats are already DDP-reduced, so the scalar is identical on every rank
        pl_module.log(self.key, slope, rank_zero_only=True)
        pl_module.log(self.key + "_intercept", intercept, rank_zero_only=True)
