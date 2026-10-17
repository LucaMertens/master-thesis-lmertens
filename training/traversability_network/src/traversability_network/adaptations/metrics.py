"""Observation-only mu calibration metrics (per-bin MAE, mean predicted mu, confusion, AUROC)."""
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanMetric

from . import mission_metrics
from .confusion_plot import bin_recalls
from .mission_metrics import RISKY_THRESHOLD

# Reported per mission and averaged over missions
REPORTED = ("calib_slope", "auroc_risky", "recall_pm1", "false_alarm")


def gather_sum(pl_module, tensor):
    """Sum a histogram buffer over ranks (all_gather only adds a world dim under DDP)."""
    gathered = pl_module.all_gather(tensor)
    return gathered.sum(0) if gathered.dim() > tensor.dim() else gathered


@torch.no_grad()
def mu_spread(trav_map):
    """Spatial std of predicted mu (B, 2, X, Y); ~0 means collapse to a constant."""
    return trav_map[:, 0].std(dim=(1, 2)).mean()


def auroc_from_hist(hist):
    """AUROC of risky mu (true < RISKY_THRESHOLD) vs the rest from a score histogram; ties count half."""
    pos, neg = hist[0].flip(0), hist[1].flip(0)  # ascending score = descending mu
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return None
    below = torch.cumsum(neg, 0) - neg
    return ((below + 0.5 * neg) * pos).sum() / (n_pos * n_neg)


def score_range(hist):
    """Predicted-mu p01, p50 and share of cells below the risky cut (AUROC ignores absolute level)."""
    total = hist.sum(0)
    n = total.sum()
    if n == 0:
        return None
    nb = hist.shape[1]
    cum = torch.cumsum(total, 0) / n
    q = torch.tensor([0.01, 0.5], device=cum.device, dtype=cum.dtype)
    p01, p50 = ((torch.searchsorted(cum, q).float() + 0.5) / nb).tolist()
    return p01, p50, float(total[: int(RISKY_THRESHOLD * nb)].sum() / n)


@torch.no_grad()
def depth_mae(grid_bounds, pred_depth, depth_target, depth_mask):
    """Masked MAE in metres between argmax-bin and target-bin depth."""
    db = grid_bounds["dbound"]
    pred_m = pred_depth.argmax(1).float() * db[2] + db[0]
    target_m = depth_target.float() * db[2] + db[0]
    mask = depth_mask > 0
    if not mask.any():
        return torch.zeros((), device=pred_depth.device)
    return (pred_m - target_m).abs()[mask].mean()


class MuBinMetrics(nn.Module):
    """Per-mu-bin eval. calib/* counts all in-grid path cells, calib_vis/* only cells with loss weight."""

    def __init__(self, n_bins=10, n_score_bins=256, mission_names=()):
        super().__init__()
        self.n_mu_bins = n_bins
        self.mu_bin_width = 1.0 / n_bins
        self.mission_names = list(mission_names)
        self.val_mae = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.val_calib = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.train_mae = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.train_calib = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.val_mae_vis = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.val_calib_vis = nn.ModuleList([MeanMetric() for _ in range(n_bins)])
        self.register_buffer(
            "val_confusion", torch.zeros(n_bins, n_bins), persistent=False
        )
        # (positives, negatives) over predicted-mu bins
        self.register_buffer(
            "val_scores", torch.zeros(2, n_score_bins), persistent=False
        )
        self.register_buffer(
            "train_scores", torch.zeros(2, n_score_bins), persistent=False
        )
        # ADAPTATION: train-side confusion
        self.register_buffer(
            "train_confusion", torch.zeros(n_bins, n_bins), persistent=False
        )
        # ADAPTATION: per-validation-mission tallies (see mission_metrics.score_hist).
        # pred_sum keeps the exact prediction so the slope matches the offline one.
        m = len(self.mission_names)
        self.register_buffer(
            "mission_confusion", torch.zeros(m, n_bins, n_bins), persistent=False
        )
        self.register_buffer("mission_pred_sum", torch.zeros(m, n_bins), persistent=False)
        self.register_buffer(
            "mission_scores", torch.zeros(m, 2, n_score_bins), persistent=False
        )

    @torch.no_grad()
    def _accumulate(self, pl_module, states, trav_map, target_trav, mae_metrics=None,
                    calib_metrics=None, weights=None, confusion=None, scores=None,
                    mission_id=None):
        """Accumulate per-bin MAE and mean predicted mu; optional gating, confusion, score and per-mission tallies."""
        pred, mask = pl_module._sample_path(states, trav_map)
        mask = mask.bool()
        if weights is not None:
            mask = mask & (weights[..., 0] > 0)
        pred_mu = pred[..., 0][mask]
        true_mu = target_trav[..., 0][mask]
        if mission_id is not None and self.mission_confusion.numel() and true_mu.numel():
            self._accumulate_missions(
                pred_mu, true_mu, mission_id[:, None].expand_as(mask)[mask]
            )
        if mae_metrics is not None:
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
        if scores is not None and true_mu.numel():
            sb = (pred_mu * scores.shape[1]).long().clamp_(0, scores.shape[1] - 1)
            low = true_mu < RISKY_THRESHOLD
            ones = torch.ones_like(sb, dtype=scores.dtype)
            scores[0].index_add_(0, sb[low], ones[low])
            scores[1].index_add_(0, sb[~low], ones[~low])

    @torch.no_grad()
    def _accumulate_missions(self, pred_mu, true_mu, gid):
        """Per-mission confusion, summed prediction per true bin, and score histogram."""
        nb, n_score = self.n_mu_bins, self.mission_scores.shape[-1]
        tb = (true_mu / self.mu_bin_width).long().clamp_(0, nb - 1)
        pb = (pred_mu / self.mu_bin_width).long().clamp_(0, nb - 1)
        ones = torch.ones_like(pred_mu, dtype=self.mission_confusion.dtype)
        self.mission_confusion.view(-1).index_add_(0, (gid * nb + tb) * nb + pb, ones)
        self.mission_pred_sum.view(-1).index_add_(
            0, gid * nb + tb, pred_mu.to(self.mission_pred_sum.dtype)
        )
        sb = (pred_mu * n_score).long().clamp_(0, n_score - 1)
        # row 0 = risky, row 1 = crossable
        crossable = (true_mu >= RISKY_THRESHOLD).long()
        self.mission_scores.view(-1).index_add_(0, (gid * 2 + crossable) * n_score + sb, ones)

    def update_train(self, pl_module, states, trav_map, target_trav, trav_weights):
        self._accumulate(pl_module, states, trav_map, target_trav, self.train_mae,
                         self.train_calib)
        # gated like valid
        self._accumulate(pl_module, states, trav_map, target_trav,
                         weights=trav_weights, scores=self.train_scores,
                         confusion=self.train_confusion)

    def update_valid(self, pl_module, states, trav_map, target_trav, trav_weights,
                     mission_id=None):
        self._accumulate(pl_module, states, trav_map, target_trav, self.val_mae, self.val_calib)
        self._accumulate(pl_module, states, trav_map, target_trav, self.val_mae_vis,
                         self.val_calib_vis, weights=trav_weights,
                         confusion=self.val_confusion, scores=self.val_scores,
                         mission_id=mission_id)

    def _log_bins(self, pl_module, prefix, mae_metrics, calib_metrics):
        """Log per-bin MAE and mean predicted mu; returns (centers, means, maes)."""
        centers, means, maes = [], [], []
        for b in range(self.n_mu_bins):
            lo = b * self.mu_bin_width
            name = f"{lo:.1f}-{lo + self.mu_bin_width:.1f}"
            # compute() is collective
            mae = mae_metrics[b].compute()
            calib = calib_metrics[b].compute()
            mae_metrics[b].reset()
            calib_metrics[b].reset()
            # empty bins give NaN
            seen = not torch.isnan(calib)
            if seen:
                pl_module.log(f"{prefix}/mae_mu_{name}", mae, sync_dist=True)
                pl_module.log(f"{prefix}/mu_{name}", calib, sync_dist=True)
            centers.append(lo + self.mu_bin_width / 2)
            means.append(float(calib) if seen else None)
            maes.append(float(mae) if seen else None)
        return centers, means, maes

    def _log_auroc(self, pl_module, prefix, scores):
        """Log low-vs-rest AUROC and predicted-mu range, then reset the histogram."""
        gathered = gather_sum(pl_module, scores)
        auroc = auroc_from_hist(gathered)
        rng = score_range(gathered)
        scores.zero_()
        if auroc is not None:
            pl_module.log(f"{prefix}/auroc_low", auroc, sync_dist=False)
        if rng is not None:
            p01, p50, frac_low = rng
            pl_module.log(f"{prefix}/pred_mu_p01", p01, sync_dist=False)
            pl_module.log(f"{prefix}/pred_mu_p50", p50, sync_dist=False)
            pl_module.log(f"{prefix}/pred_frac_low", frac_low, sync_dist=False)

    def log_train(self, pl_module):
        """Train-side twin of the valid block, logged from the validation hook."""
        _, means, _ = self._log_bins(pl_module, "calib_train", self.train_mae, self.train_calib)
        self._log_auroc(pl_module, "train", self.train_scores)
        counts = gather_sum(pl_module, self.train_confusion)
        exact, within1 = bin_recalls(counts.cpu().numpy())
        pl_module.log("train/recall_exact_bin", exact, sync_dist=False)
        pl_module.log("train/recall_pm1_bin", within1, sync_dist=False)
        self._log_confusion_image(pl_module, means, counts, "calib_train/confusion")
        self.train_confusion.zero_()

    def reset_confusion(self):
        self.val_confusion.zero_()
        self.val_scores.zero_()
        self.mission_confusion.zero_()
        self.mission_pred_sum.zero_()
        self.mission_scores.zero_()

    def log_missions(self, pl_module):
        """Reported metrics per validation mission and averaged (same rule as offline scoring).

        The slope is always logged since it is the checkpoint monitor; NaN reads as 0."""
        if not self.mission_confusion.numel():
            return
        counts = gather_sum(pl_module, self.mission_confusion).cpu().numpy()
        pred_sum = gather_sum(pl_module, self.mission_pred_sum).cpu().numpy()
        scores = gather_sum(pl_module, self.mission_scores).cpu().numpy()
        per = {
            name: mission_metrics.score_hist(counts[i], pred_sum[i], scores[i])
            for i, name in enumerate(self.mission_names)
            if counts[i].sum() > 0
        }
        mean = mission_metrics.average(per)
        for name, s in per.items():
            for key in REPORTED:
                if not np.isnan(s[key]):
                    pl_module.log(f"valid_mission/{name}/{key}", s[key], sync_dist=False)
        slope = mean["calib_slope"]
        pl_module.log("valid_mission/calib_slope",
                      0.0 if np.isnan(slope) else slope, sync_dist=False)
        for key in ("auroc_risky", "recall_pm1", "false_alarm"):
            if not np.isnan(mean[key]):
                pl_module.log(f"valid_mission/{key}", mean[key], sync_dist=False)
        pl_module.log("valid_mission/n_missions_scored",
                      float(mean["n_missions_scored"]), sync_dist=False)

    def on_validation_epoch_end(self, pl_module):
        """Log per-sub-bin MAE, mean predicted mu, and the confusion-matrix image."""
        self.log_train(pl_module)
        self._log_bins(pl_module, "calib", self.val_mae, self.val_calib)
        _, means_vis, vis_maes = self._log_bins(pl_module, "calib_vis", self.val_mae_vis, self.val_calib_vis)
        # macro-average over mu bins, so rare low-mu bins count as much as common ones
        seen = [m for m in vis_maes if m is not None]
        balanced_mae = sum(seen) / len(seen) if seen else None
        if balanced_mae is not None:
            pl_module.log("valid/balanced_mae", balanced_mae, sync_dist=False)
        # risky recall from the visible-cell confusion: ~0 = collapsed
        counts = gather_sum(pl_module, self.val_confusion)
        cut = int(RISKY_THRESHOLD * self.n_mu_bins)
        risky_true = counts[:cut, :].sum()
        risky_recall = (counts[:cut, :cut].sum() / risky_true) if risky_true > 0 else torch.zeros((), device=counts.device)
        pl_module.log("valid/risky_recall", risky_recall, sync_dist=False)
        # false alarm: crossable cells called risky
        rest_true = counts[cut:, :].sum()
        if rest_true > 0:
            pl_module.log("valid/false_alarm", counts[cut:, :cut].sum() / rest_true,
                          sync_dist=False)
        exact, within1 = bin_recalls(counts.cpu().numpy())
        pl_module.log("valid/recall_exact_bin", exact, sync_dist=False)
        pl_module.log("valid/recall_pm1_bin", within1, sync_dist=False)
        self._log_auroc(pl_module, "valid", self.val_scores)
        # ADAPTATION: per-mission average
        self.log_missions(pl_module)
        # calibration slope of mean predicted mu vs bin center (~0.55+ healthy, ~0.2 compressed).
        # Always logged (checkpoint monitor); <2 occupied bins reads as 0.
        pts = [(b * self.mu_bin_width + self.mu_bin_width / 2, m)
               for b, m in enumerate(means_vis) if m is not None]
        slope = torch.zeros(())
        if len(pts) > 1:
            xs = torch.tensor([p[0] for p in pts])
            ys = torch.tensor([p[1] for p in pts])
            slope = ((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()).square().sum() + 1e-9)
        pl_module.log("valid/calib_slope", slope, sync_dist=False)
        self._log_confusion_image(pl_module, means_vis)

    def _log_confusion_image(self, pl_module, means_vis, counts=None, key="calib/confusion"):
        # all_gather is collective, only rank 0 renders
        if counts is None:
            counts = gather_sum(pl_module, self.val_confusion)
        if not pl_module.trainer.is_global_zero or not isinstance(pl_module.logger, WandbLogger):
            return
        if counts.sum() == 0:
            return
        import wandb
        from .confusion_plot import render_confusion, training_title
        import matplotlib.pyplot as plt

        configs = pl_module.hparams.configs
        run = pl_module.logger.experiment
        architecture, train_set, valid_set = training_title(
            configs, run.name, pl_module.current_epoch
        )
        on_train = key.startswith("calib_train")
        fig = render_confusion(
            counts.cpu().numpy(), means_vis, architecture=architecture,
            train_set=train_set, valid_set=None if on_train else valid_set,
        )
        pl_module.logger.experiment.log(
            {key: wandb.Image(fig)}, step=pl_module.trainer.global_step
        )
        plt.close(fig)
