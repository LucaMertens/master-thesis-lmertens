"""Observation-only mu calibration metrics (per-bin MAE, mean predicted mu, confusion matrix), delegated to by the trainer's Lightning hooks."""
import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanMetric


@torch.no_grad()
def mu_spread(trav_map):
    """Spatial std of predicted mu (B, 2, X, Y). Collapse to a constant output
    shows up here as ~0, epochs before the per-bin calibration flattens."""
    return trav_map[:, 0].std(dim=(1, 2)).mean()


@torch.no_grad()
def depth_mae(grid_bounds, pred_depth, depth_target, depth_mask):
    """Masked MAE in metres between argmax-bin and target-bin depth. The depth CE
    can rise from overconfidence alone; this says whether depth itself drifts."""
    db = grid_bounds["dbound"]
    pred_m = pred_depth.argmax(1).float() * db[2] + db[0]
    target_m = depth_target.float() * db[2] + db[0]
    mask = depth_mask > 0
    if not mask.any():
        return torch.zeros((), device=pred_depth.device)
    return (pred_m - target_m).abs()[mask].mean()


class MuBinMetrics(nn.Module):
    """Graded low-mu eval over [0.0,0.1) ... [0.9,1.0). torchmetrics handles DDP
    reduction. calib/* counts every in-grid path cell, including ones no input
    frame can see; calib_vis/* restricts to cells with nonzero loss weight (visible
    and observable). Train and valid keep separate bins: memorization (train sharp,
    valid flat) and global collapse (both flat) are different failures."""

    def __init__(self, n_bins=10):
        super().__init__()
        self.n_mu_bins = n_bins
        self.mu_bin_width = 1.0 / n_bins
        self.val_mae = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.val_calib = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.train_mae = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.train_calib = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.val_mae_vis = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.val_calib_vis = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.register_buffer(
            "val_confusion", torch.zeros(n_bins, n_bins), persistent=False
        )

    @torch.no_grad()
    def _accumulate(self, pl_module, states, trav_map, target_trav, mae_metrics,
                    calib_metrics, weights=None, confusion=None):
        """Accumulate per-true-mu-bin MAE and mean predicted mu over valid cells.
        With ``weights``, only cells carrying loss weight count (visibility and
        observability gates). With ``confusion``, also tally the 10x10 bin counts."""
        pred, mask = pl_module._sample_path(states, trav_map)
        mask = mask.bool()
        if weights is not None:
            mask = mask & (weights[..., 0] > 0)
        pred_mu = pred[..., 0][mask]
        true_mu = target_trav[..., 0][mask]
        for b in range(self.n_mu_bins):
            lo = b * self.mu_bin_width
            sel = (true_mu >= lo) & (true_mu < lo + self.mu_bin_width)
            if sel.any():
                mae_metrics[b].update((pred_mu[sel] - true_mu[sel]).abs())
                calib_metrics[b].update(pred_mu[sel])
        if confusion is not None and true_mu.numel():
            tb = (true_mu / self.mu_bin_width).long().clamp_(0, self.n_mu_bins - 1)
            pb = (pred_mu / self.mu_bin_width).long().clamp_(0, self.n_mu_bins - 1)
            idx = tb * self.n_mu_bins + pb
            confusion.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=confusion.dtype))

    def update_train(self, pl_module, states, trav_map, target_trav):
        self._accumulate(pl_module, states, trav_map, target_trav, self.train_mae, self.train_calib)

    def update_valid(self, pl_module, states, trav_map, target_trav, trav_weights):
        self._accumulate(pl_module, states, trav_map, target_trav, self.val_mae, self.val_calib)
        self._accumulate(pl_module, states, trav_map, target_trav, self.val_mae_vis,
                         self.val_calib_vis, weights=trav_weights, confusion=self.val_confusion)

    def _log_bins(self, pl_module, prefix, mae_metrics, calib_metrics):
        """Log per-bin MAE and mean predicted mu; returns (centers, means, maes)."""
        centers, means, maes = [], [], []
        for b in range(self.n_mu_bins):
            lo = b * self.mu_bin_width
            name = f"{lo:.1f}-{lo + self.mu_bin_width:.1f}"
            # compute() is collective: call for every bin on every rank
            mae = mae_metrics[b].compute()
            calib = calib_metrics[b].compute()
            mae_metrics[b].reset()
            calib_metrics[b].reset()
            # empty bins compute() to NaN (same on every rank, so logging stays consistent)
            seen = not torch.isnan(calib)
            if seen:
                pl_module.log(f"{prefix}/mae_mu_{name}", mae, sync_dist=True)
                pl_module.log(f"{prefix}/mu_{name}", calib, sync_dist=True)
            centers.append(lo + self.mu_bin_width / 2)
            means.append(float(calib) if seen else None)
            maes.append(float(mae) if seen else None)
        return centers, means, maes

    def on_train_epoch_end(self, pl_module):
        self._log_bins(pl_module, "calib_train", self.train_mae, self.train_calib)

    def reset_confusion(self):
        self.val_confusion.zero_()

    def on_validation_epoch_end(self, pl_module):
        """Log per-sub-bin MAE, mean predicted mu, and the confusion-matrix image."""
        self._log_bins(pl_module, "calib", self.val_mae, self.val_calib)
        _, means_vis, vis_maes = self._log_bins(pl_module, "calib_vis", self.val_mae_vis, self.val_calib_vis)
        # Macro-average over mu bins (equal weight per bin, not per sample), so the
        # rare low-mu bins count as much as the common high-mu mass. trav_loss and
        # micro MAE stay flat straight through mean collapse; this doesn't.
        seen = [m for m in vis_maes if m is not None]
        balanced_mae = sum(seen) / len(seen) if seen else None
        if balanced_mae is not None:
            pl_module.log("valid/balanced_mae", balanced_mae, sync_dist=False)
        self._log_confusion_image(pl_module, means_vis, balanced_mae)

    def _log_confusion_image(self, pl_module, means_vis, balanced_mae):
        # all_gather is collective: every rank must call it, then only rank 0 renders.
        counts = pl_module.all_gather(self.val_confusion).sum(0)
        if not pl_module.trainer.is_global_zero or not isinstance(pl_module.logger, WandbLogger):
            return
        if counts.sum() == 0:
            return
        import wandb
        from .confusion_plot import render_confusion
        import matplotlib.pyplot as plt

        fig = render_confusion(
            counts.cpu().numpy(), means_vis, balanced_mae,
            title=f"Valid mu confusion (visible cells) - epoch {pl_module.current_epoch}",
        )
        pl_module.logger.experiment.log(
            {"calib/confusion": wandb.Image(fig)}, step=pl_module.trainer.global_step
        )
        plt.close(fig)
