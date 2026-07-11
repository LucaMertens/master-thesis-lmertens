# Taken from wayfaster repository (https://github.com/matval/wayfaster)
# Modifications: fixed import of TravNet for local package layout.
import math
import os
import pathlib
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MeanMetric

from .models.traversability_net import TravNet
from ..overlay import colorize_mu, overlay_bev_on_image, overlay_path_on_image

# Path overlay colors (RGB): red line, green start (robot, t=0), white end (horizon).
PATH_LINE = (255, 0, 0)
PATH_START = (0, 255, 0)
PATH_END = (255, 255, 255)

class TrainingModule(pl.LightningModule):
    """
    Training module for the TravNet model using PyTorch Lightning.
    """
    def __init__(self, configs):
        """
        Initialize the TrainingModule.

        Args:
            configs (object): Configuration object containing training parameters.
        """
        super().__init__()
        # Save hyperparamters to hparams.yaml
        self.save_hyperparameters()
        
        # Get training params
        self.eps = 1e-6
        gamma = torch.tensor([configs.TRAINING.GAMMA**i for i in range(configs.TRAINING.HORIZON)])
        # Harvested low-mu extras sit past the horizon slots; they are visible
        # in-grid cells, so no temporal discount.
        self.path_horizon = configs.TRAINING.HORIZON
        harvest_extra = getattr(configs.DATASET, "HARVEST_LOWMU_EXTRA", 0)
        if harvest_extra:
            gamma = torch.cat([gamma, torch.ones(harvest_extra)])
        self.gamma = nn.Parameter(gamma, requires_grad=False)
        # Mu loss selector. LOSS_TYPE, when set, is authoritative; otherwise the loss
        # is inferred from the legacy MU_BINS/BMC_SIGMA_INIT/EVIDENTIAL flags.
        mu_bins = getattr(configs.MODEL, "MU_BINS", 0)
        bmc_init = getattr(configs.TRAINING, "BMC_SIGMA_INIT", 0.0)
        evidential = getattr(configs.MODEL, "EVIDENTIAL", False)
        loss_type = getattr(configs.TRAINING, "LOSS_TYPE", None)
        if loss_type is not None:
            valid = {"regression", "bmc", "binned_ce", "evidential"}
            assert loss_type in valid, f"LOSS_TYPE must be one of {valid}, got {loss_type!r}"
            if loss_type == "bmc":
                assert bmc_init > 0, "LOSS_TYPE=bmc needs TRAINING.BMC_SIGMA_INIT > 0"
            else:
                bmc_init = 0.0
            if loss_type == "binned_ce":
                assert mu_bins > 0, "LOSS_TYPE=binned_ce needs MODEL.MU_BINS > 0"
            else:
                mu_bins = 0
            evidential = loss_type == "evidential"
        # Balanced MSE (BMC) noise scale, learnable; None = smooth-L1 mu term
        self.bmc_sigma = nn.Parameter(torch.tensor(float(bmc_init))) if bmc_init > 0 else None
        self.learning_rate = configs.OPTIMIZER.LR
        self.lr_ref_batchsize = configs.OPTIMIZER.LR_REF_BATCHSIZE
        self.backbone_lr_factor = getattr(configs.OPTIMIZER, "BACKBONE_LR_FACTOR", 1.0)
        self.batchsize = configs.TRAINING.BATCHSIZE
        self.weight_decay = configs.OPTIMIZER.WEIGHT_DECAY
        self.depth_weight = configs.TRAINING.DEPTH_WEIGHT
        self.overpred_weight = getattr(configs.TRAINING, "TRAV_OVERPRED_WEIGHT", 1.0)
        self.mu_head_bins = mu_bins
        # Evidential (NIG) mu head: replaces the mu smooth-L1/binned-CE/BMC term with
        # the Deep Evidential Regression loss (Amini et al. 2020); nu is unaffected.
        self.evidential = evidential
        self.evidential_lambda = getattr(configs.TRAINING, "EVIDENTIAL_LAMBDA", 0.01)
        assert not (self.evidential and self.mu_head_bins), "EVIDENTIAL and MU_BINS are mutually exclusive"
        assert not (self.evidential and self.bmc_sigma is not None), "EVIDENTIAL and BMC_SIGMA_INIT are mutually exclusive"
        self.epochs = configs.TRAINING.EPOCHS
        
        # Depth training
        self.train_depth = configs.MODEL.TRAIN_DEPTH
        
        # Depth prediction
        self.predict_depth = configs.MODEL.PREDICT_DEPTH
        
        # Mapping Model
        self.model = TravNet(
            configs.MODEL.GRID_BOUNDS,
            configs.MODEL.INPUT_SIZE,
            downsample=configs.MODEL.DOWNSAMPLE,
            image_dim=configs.MODEL.LATENT_DIM,
            temporal_length=configs.MODEL.TIME_LENGTH,
            predict_depth=configs.MODEL.PREDICT_DEPTH,
            fuse_pcloud=configs.MODEL.FUSE_PCLOUD,
            voxel_downsample=configs.MODEL.VOXEL_DEPTH_DOWNSAMPLE,
            mu_bins=self.mu_head_bins,
            evidential=self.evidential,
        )

        # Convert to channels_last to avoid DDP bucket stride mismatch warnings
        # when torch.compile enables cuDNN channels-last optimizations.
        # We only apply this to specific submodules to avoid RuntimeError on 3D convolutions.
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                m.to(memory_format=torch.channels_last)
            elif isinstance(m, (nn.Conv3d, nn.BatchNorm3d)):
                m.to(memory_format=torch.channels_last_3d)

        # Compile the network (fixed shapes -> big fused-kernel win). voxelize() is
        # @torch.compiler.disable'd so its data-dependent scatter stays in eager.
        if configs.TRAINING.COMPILE:
            import torch._dynamo as _dynamo
            # Increase recompile limit because BasicBlock is reused many times (encoder + decoder)
            # with different input shapes/strides, and across train/val (grad enabled/disabled).
            _dynamo.config.recompile_limit = 64
            self.model = torch.compile(self.model)

        # Set grid bounds
        self.grid_bounds = configs.MODEL.GRID_BOUNDS
        self.map_size = (
            int((self.grid_bounds['xbound'][1] - self.grid_bounds['xbound'][0])/self.grid_bounds['xbound'][2]),
            int((self.grid_bounds['ybound'][1] - self.grid_bounds['ybound'][0])/self.grid_bounds['ybound'][2]))

        self.map_origin = (
            int((self.grid_bounds['xbound'][1])/self.grid_bounds['xbound'][2]),
            int((self.grid_bounds['ybound'][1])/self.grid_bounds['ybound'][2]))

        self.map_resolution = (
            self.grid_bounds['xbound'][2],
            self.grid_bounds['ybound'][2])

        # Pointcloud dropout (applied GPU-side, train only, by zeroing whole depth windows)
        self.pcloud_dropout = configs.AUGMENTATIONS.PCLOUD_DROPOUT

        # Visualization rate
        self.vis_interval = configs.TRAINING.VIS_INTERVAL
        self.vis_batch_size = getattr(configs.TRAINING, "VIS_BATCH_SIZE", 8)
        # Optional debug panels (see visualize)
        self.debug_vis_labels = getattr(configs.TRAINING, "DEBUG_VIS_LABELS", False)
        self.debug_vis_geom = getattr(configs.TRAINING, "DEBUG_VIS_GEOM", False)
        self.debug_vis_depth = getattr(configs.TRAINING, "DEBUG_VIS_DEPTH", False)

        # Graded low-mu eval: per-0.1-bin MAE and mean-predicted-mu (calibration)
        # over [0.0,0.1) ... [0.9,1.0). torchmetrics handles DDP reduction.
        self.n_mu_bins = 10
        self.mu_bin_width = 0.1
        self.val_mae = nn.ModuleList([MeanMetric() for _ in range(self.n_mu_bins)])
        self.val_calib = nn.ModuleList([MeanMetric() for _ in range(self.n_mu_bins)])
        # Same per-bin metrics on train batches: memorization (train sharp, valid
        # flat) and global collapse (both flat) are different failures.
        self.train_mae = nn.ModuleList([MeanMetric() for _ in range(self.n_mu_bins)])
        self.train_calib = nn.ModuleList([MeanMetric() for _ in range(self.n_mu_bins)])
        # calib/* counts every in-grid path cell, including ones no input frame can
        # see; calib_vis/* restricts to cells with nonzero loss weight (visible and
        # observable), i.e. the cells the network can actually be expected to predict.
        self.val_mae_vis = nn.ModuleList([MeanMetric() for _ in range(self.n_mu_bins)])
        self.val_calib_vis = nn.ModuleList([MeanMetric() for _ in range(self.n_mu_bins)])

    def prepare_data(self):
        """Called exactly once per node by PyTorch Lightning to prepare data."""
        configs = self.hparams.configs if hasattr(self.hparams, "configs") else getattr(self, "configs", None)
        if not configs:
            return

        missions = list(dict.fromkeys(
            configs.DATASET.TRAIN_MISSIONS + 
            configs.DATASET.VALID_MISSIONS + 
            getattr(configs.DATASET, "TEST_MISSIONS", [])
        ))

        if configs.DATASET.RESIZED_ROOT:
            from grandtour_extraction.resize_images import resize_mission
            width, height = configs.MODEL.INPUT_SIZE
            for mission in missions:
                resize_mission(
                    pathlib.Path(configs.DATASET.DATA_ROOT) / mission,
                    pathlib.Path(configs.DATASET.RESIZED_ROOT),
                    width, height, n_workers=os.cpu_count(), png_compression=1, force=False,
                )

        if getattr(configs.DATASET, "DEPTH_CACHE_ROOT", None):
            from grandtour_extraction.cache_depth import cache_mission
            for mission in missions:
                cache_mission(
                    pathlib.Path(configs.DATASET.DATA_ROOT) / mission,
                    pathlib.Path(configs.DATASET.DEPTH_CACHE_ROOT),
                    n_workers=os.cpu_count(), force=False,
                )

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): Batch of data containing images, point clouds, intrinsics, extrinsics, paths, targets, and masks.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        # Get data
        color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics, path, target_trav, trav_weights, depth_target, depth_mask = batch

        # Pointcloud dropout: zero whole depth windows so the GPU builds an empty grid
        depth_raw = depth_voxel  # undropped, for the viz panels
        if self.pcloud_dropout > 0:
            drop = torch.rand(depth_voxel.shape[0], device=depth_voxel.device) < self.pcloud_dropout
            if drop.any():
                depth_voxel = depth_voxel.clone()
                depth_voxel[drop] = 0

        # Forward pass
        trav_map, mu_logits, pred_depth, debug = self.model(color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics, depth_target)

        # Calculate traversability loss
        trav_loss, _ = self.trav_criterion(path, trav_map, target_trav, trav_weights, mu_logits)

        # Update graded low-mu metrics on train batches (logged at epoch end)
        self._update_mu_metrics(path, trav_map, target_trav, self.train_mae, self.train_calib)

        # Calculate depth classification loss
        depth_target = depth_target.view(-1, *depth_target.shape[2:])
        depth_mask = depth_mask.view(-1, *depth_mask.shape[2:])
        depth_loss = self.depth_criterion(pred_depth, depth_target, depth_mask)

        if self.train_depth:
            loss = trav_loss + self.depth_weight*depth_loss
        else:
            loss = trav_loss

        # Visualize results
        if (batch_idx % self.vis_interval) == 0:
            pcloud = self._voxelize_for_vis(color_img, depth_voxel, intrinsics_full, extrinsics)
            self.visualize(color_img, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, path, inv_intrinsics, extrinsics, target_trav, trav_weights, mu_logits, depth_voxel=depth_raw, intrinsics_full=intrinsics_full, prefix='train')

        if batch_idx % 1000 == 0:
            self.print(f"[epoch {self.current_epoch} step {batch_idx}] loss={loss.item():.4f} trav={trav_loss.item():.4f} depth={depth_loss.item():.4f}")

        # rank-local values: sync_dist would add three all-reduces every step
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        self.log("train/trav_loss", trav_loss, on_step=True, on_epoch=False)
        self.log("train/depth_loss", depth_loss, on_step=True, on_epoch=False)
        self.log("train/depth_mae", self._depth_mae(pred_depth, depth_target, depth_mask), on_step=True, on_epoch=False)
        self.log("train/mu_spread", self._mu_spread(trav_map), on_step=True, on_epoch=False)

        return loss

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, '_batch_start_time'):
            step_time = time.time() - self._batch_start_time
            self.log("perf/time_per_step", step_time, on_step=True, on_epoch=False)

    def on_train_epoch_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if hasattr(self, '_epoch_start_time'):
            epoch_time = time.time() - self._epoch_start_time
            self.log("perf/time_per_epoch", epoch_time, on_step=False, on_epoch=True, sync_dist=True)
        self._log_mu_bins("calib_train", self.train_mae, self.train_calib)

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): Batch of data containing images, point clouds, intrinsics, extrinsics, paths, targets, and masks.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        # Get data
        color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics, path, target_trav, trav_weights, depth_target, depth_mask = batch

        # Forward pass
        trav_map, mu_logits, pred_depth, debug = self.model(color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics, depth_target)

        # Calculate traversability loss
        trav_loss, trav_error = self.trav_criterion(path, trav_map, target_trav, trav_weights, mu_logits)

        # Update graded low-mu metrics (per-0.1-bin MAE + calibration)
        self._update_mu_metrics(path, trav_map, target_trav, self.val_mae, self.val_calib)
        self._update_mu_metrics(path, trav_map, target_trav, self.val_mae_vis, self.val_calib_vis, weights=trav_weights)

        # Calculate depth classification loss
        depth_target = depth_target.view(-1, *depth_target.shape[2:])
        depth_mask = depth_mask.view(-1, *depth_mask.shape[2:])
        depth_loss = self.depth_criterion(pred_depth, depth_target, depth_mask)

        if self.train_depth:
            loss = trav_loss + self.depth_weight*depth_loss
        else:
            loss = trav_loss

        # Visualize a fixed set of batches spread across the whole shard
        if batch_idx in self._val_vis_batches:
            pcloud = self._voxelize_for_vis(color_img, depth_voxel, intrinsics_full, extrinsics)
            self.visualize(color_img, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, path, inv_intrinsics, extrinsics, target_trav, trav_weights, mu_logits, depth_voxel=depth_voxel, intrinsics_full=intrinsics_full, prefix='valid')

        self.log("valid/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/trav_loss", trav_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/trav_error", trav_error, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/depth_loss", depth_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/depth_mae", self._depth_mae(pred_depth, depth_target, depth_mask), on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/mu_spread", self._mu_spread(trav_map), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_start(self):
        # VIS_INTERVAL can exceed the shard's batch count; the old modulo check then
        # only ever showed batch 0 (the first windows of the first mission). Spread a
        # fixed handful of panels across the shard instead.
        n = self.trainer.num_val_batches[0]
        if isinstance(n, (int, float)) and np.isfinite(n) and n > 0:
            self._val_vis_batches = {int(i) for i in np.linspace(0, int(n) - 1, min(4, int(n)))}
        else:
            self._val_vis_batches = {0}

    @torch.no_grad()
    def _update_mu_metrics(self, states, trav_map, target_trav, mae_metrics, calib_metrics, weights=None):
        """Accumulate per-true-mu-bin MAE and mean predicted mu over valid cells.
        With ``weights``, only cells carrying loss weight count (visibility and
        observability gates)."""
        pred, mask = self._sample_path(states, trav_map)
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

    def _log_mu_bins(self, prefix, mae_metrics, calib_metrics):
        """Log per-bin MAE and mean predicted mu; returns (centers, means, maes) for curves."""
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
                self.log(f"{prefix}/mae_mu_{name}", mae, sync_dist=True)
                self.log(f"{prefix}/mu_{name}", calib, sync_dist=True)
            centers.append(lo + self.mu_bin_width / 2)
            means.append(float(calib) if seen else None)
            maes.append(float(mae) if seen else None)
        return centers, means, maes

    def on_validation_epoch_end(self):
        """Log per-sub-bin MAE, mean predicted mu, and a calibration curve."""
        centers, means, _ = self._log_mu_bins("calib", self.val_mae, self.val_calib)
        _, _, vis_maes = self._log_mu_bins("calib_vis", self.val_mae_vis, self.val_calib_vis)
        self._log_calibration_curve(centers, means)
        # Macro-average over mu bins (equal weight per bin, not per sample), so the
        # rare low-mu bins count as much as the common high-mu mass. trav_loss and
        # micro MAE stay flat straight through mean collapse; this doesn't.
        seen = [m for m in vis_maes if m is not None]
        if seen:
            self.log("valid/balanced_mae", sum(seen) / len(seen), sync_dist=False)

    def _log_calibration_curve(self, centers, means):
        if not self.trainer.is_global_zero or not isinstance(self.logger, WandbLogger):
            return
        import wandb
        rows = [[c, m] for c, m in zip(centers, means) if m is not None]
        if not rows:
            return
        table = wandb.Table(data=rows, columns=["true_mu_bin", "mean_pred_mu"])
        self.logger.experiment.log(
            {
                "calib/table": table,
                "calib/curve": wandb.plot.line(
                    table, "true_mu_bin", "mean_pred_mu",
                    title="Calibration: mean predicted mu vs true mu bin",
                ),
            },
            step=self.trainer.global_step,
        )

    def configure_optimizers(self):
        """
        Configure the optimizers and learning rate scheduler.

        Returns:
            list: List containing the optimizer and the scheduler.
        """
        # Sqrt LR scaling for AdamW: scale with effective batch relative to the tuning reference.
        lr_scale = (self.batchsize * self.trainer.world_size / self.lr_ref_batchsize) ** 0.5
        lr = self.learning_rate * lr_scale
        # Pretrained ResNet stages (encoder.backbone / encoder.layer*) at a reduced LR;
        # the freshly initialized heads and BEV stack at the full LR.
        pretrained = [
            p for n, p in self.model.named_parameters()
            if "encoder.backbone" in n or "encoder.layer" in n
        ]
        pretrained_ids = {id(p) for p in pretrained}
        rest = [p for p in self.model.parameters() if id(p) not in pretrained_ids]
        groups = [
            {"params": rest, "lr": lr},
            {"params": pretrained, "lr": lr * self.backbone_lr_factor},
        ]
        if self.bmc_sigma is not None:
            # single scalar; the BMC reference implementation trains it at ~1e-2
            groups.append({"params": [self.bmc_sigma], "lr": 1e-2, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)

        # Learning rate scheduler: one full cosine decay over the whole run (T_max must
        # match EPOCHS, else the LR rises back toward its max in the second half).
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5 * lr_scale)
        
        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        # bf16-mixed has no GradScaler, so a single batch with non-finite gradients
        # would permanently corrupt the weights and silently stall training (global_step
        # stops advancing). Skip such updates, like fp16's GradScaler does for free.
        # Fused norms + one .item(): a per-parameter isfinite loop costs a GPU sync
        # per parameter (~15 ms/step for 209 params vs ~1 ms for this).
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        finite = (
            len(grads) == 0
            or torch.isfinite(torch.stack(torch._foreach_norm(grads))).all().item()
        )
        if not finite:
            self._nonfinite_skips = getattr(self, "_nonfinite_skips", 0) + 1
            if self._nonfinite_skips <= 5 or self._nonfinite_skips % 50 == 0:
                print(
                    f"[warn] non-finite gradients at step {self.global_step}: skipping "
                    f"update (total skipped: {self._nonfinite_skips})",
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)

    def trav_criterion(self, states, trav_map, target_trav, target_weights, mu_logits=None):
        """
        Calculate the traversability loss.

        Args:
            states (torch.Tensor): Tensor containing the states.
            trav_map (torch.Tensor): Tensor containing the traversability map.
            target_trav (torch.Tensor): Tensor containing the target traversability.
            target_weights (torch.Tensor): Tensor containing the target weights.
            mu_logits (torch.Tensor, optional): auxiliary mu tensor from TravNet.forward:
                bin logits (B, K, X, Y) with the binned head, or NIG (v, alpha, beta)
                (B, 3, X, Y) with the evidential head; unused for plain regression.

        Returns:
            tuple: Tuple containing the loss and the error.
        """
        # Sample predicted traversability along the path
        traversability, mask = self._sample_path(states, trav_map)

        # Calculate loss
        gamma = self.gamma[:traversability.shape[1]].view(1,-1,1).expand(traversability.shape[0],-1,2)
        error = F.smooth_l1_loss(traversability, target_trav, reduction='none', beta=0.5)
        # Binned mu head: cross-entropy against a soft two-hot target instead of
        # regressing the expectation; nu keeps its smooth-L1 term.
        train_error = error
        if self.mu_head_bins:
            logits, _ = self._sample_path(states, mu_logits)
            ce = -(self._mu_two_hot(target_trav[..., 0]) * logits.log_softmax(-1)).sum(-1)
            train_error = torch.stack([ce, error[..., 1]], dim=-1)
        elif self.evidential:
            aux, _ = self._sample_path(states, mu_logits)
            nig = self._evidential_loss(traversability[..., 0], target_trav[..., 0], aux)
            train_error = torch.stack([nig, error[..., 1]], dim=-1)
        # Asymmetric penalty: optimistic errors (predicting traversable where the
        # label says slippery) cost overpred_weight times more. For the binned head
        # the indicator uses the expected mu.
        if self.overpred_weight != 1.0:
            over = (traversability > target_trav).detach().to(error.dtype)
            train_error = train_error * (1 + (self.overpred_weight - 1) * over)
        weight = target_weights * gamma
        # Weighted average, not weighted sum/count: keeps the loss scale (and its
        # gradient norm) independent of how much low-mu weight mass a batch happens
        # to contain, so gradient_clip_val doesn't disproportionately clip the rare
        # high-weight batches back down to the common ones.
        if self.bmc_sigma is not None:
            # Balanced MSE replaces the mu term; nu keeps its weighted smooth-L1
            mu_loss = self._bmc_loss(
                traversability[..., 0], target_trav[..., 0], weight[..., 0] * mask
            )
            nu_weight = weight[..., 1][mask]
            nu_loss = (train_error[..., 1][mask] * nu_weight).sum() / (nu_weight.sum() + self.eps)
            loss = mu_loss + nu_loss
        else:
            weighted_error = (train_error * weight)[mask, :]
            loss = weighted_error.sum() / (weight[mask, :].sum() + self.eps)

        # Error stays smooth-L1 on (mu, nu) in both modes, comparable across runs
        error = torch.mean(error[mask,:])

        return loss, error

    def _mu_two_hot(self, mu):
        """Soft two-hot encoding of continuous mu over the K bin centers."""
        K = self.mu_head_bins
        x = (mu * K - 0.5).clamp(0, K - 1)
        lo = x.floor()
        frac = (x - lo).unsqueeze(-1)
        lo = lo.long().unsqueeze(-1)
        hi = (lo + 1).clamp(max=K - 1)
        target = torch.zeros(*mu.shape, K, device=mu.device, dtype=mu.dtype)
        target.scatter_(-1, lo, 1 - frac)
        target.scatter_add_(-1, hi, frac)
        return target

    def _evidential_loss(self, gamma, y, aux):
        """Deep Evidential Regression loss (Amini et al. 2020) for a Normal-Inverse-
        Gamma prediction (gamma, v, alpha, beta) of the mu label y. NLL marginalizes
        the unknown Gaussian mean/variance under the NIG prior; the regularizer
        penalizes high evidence (v, alpha) on samples where gamma is still wrong,
        which is what makes the evidence shrink on OOD/hard inputs instead of just
        anywhere the residual is nonzero."""
        v, alpha, beta = aux[..., 0], aux[..., 1], aux[..., 2]
        omega = 2 * beta * (1 + v)
        nll = (
            0.5 * torch.log(math.pi / v)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log((y - gamma).square() * v + omega)
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        )
        reg = (y - gamma).abs() * (2 * v + alpha)
        return nll + self.evidential_lambda * reg

    def _bmc_loss(self, pred, target, weight):
        """Balanced MSE, BMC variant (Ren et al. 2022): cross-entropy between each
        prediction and all batch targets under a learnable Gaussian noise scale.
        The batch targets serve as a Monte Carlo sample of the label distribution,
        which corrects the mean-seeking bias of plain (smooth) L1/L2 without a
        precomputed density."""
        sel = weight > 0
        pred, target, weight = pred[sel], target[sel], weight[sel]
        if pred.numel() == 0:
            return pred.sum()
        noise_var = self.bmc_sigma.square().clamp_min(1e-4)
        logits = -(pred[:, None] - target[None, :]).square() / (2 * noise_var)
        ce = F.cross_entropy(
            logits, torch.arange(len(pred), device=pred.device), reduction="none"
        )
        # detached 2*sigma^2 keeps the loss on the MSE scale as sigma adapts
        ce = ce * (2 * noise_var).detach()
        return (ce * weight).sum() / (weight.sum() + self.eps)

    def _sample_path(self, states, trav_map):
        """Sample the predicted traversability map along the path states.

        Returns (traversability (B, H, 2), mask (B, H) of in-bounds cells).
        """
        idxu = 2 * (states[...,0] - self.grid_bounds['xbound'][0]) / (self.grid_bounds['xbound'][1] - self.grid_bounds['xbound'][0]) - 1
        idxv = 2 * (states[...,1] - self.grid_bounds['ybound'][0]) / (self.grid_bounds['ybound'][1] - self.grid_bounds['ybound'][0]) - 1
        grid = torch.stack((-idxv, -idxu), -1).unsqueeze(1)
        mask = (grid[...,0] >= -1) * (grid[...,0] <= 1) * (grid[...,1] >= -1) * (grid[...,1] <= 1)

        traversability = F.grid_sample(trav_map, grid, align_corners=True)
        traversability = traversability.permute((0,2,3,1)).squeeze(1)

        return traversability, mask.squeeze(1)
    
    @torch.no_grad()
    def _depth_mae(self, pred_depth, depth_target, depth_mask):
        """Masked MAE in metres between argmax-bin and target-bin depth. The depth CE
        can rise from overconfidence alone; this says whether depth itself drifts."""
        db = self.grid_bounds['dbound']
        pred_m = pred_depth.argmax(1).float() * db[2] + db[0]
        target_m = depth_target.float() * db[2] + db[0]
        mask = depth_mask > 0
        if not mask.any():
            return torch.zeros((), device=pred_depth.device)
        return (pred_m - target_m).abs()[mask].mean()

    @torch.no_grad()
    def _mu_spread(self, trav_map):
        """Scalar measuring spatial variation of the predicted mu maps (B, 2, X, Y).
        Collapse to a constant output shows up here as ~0, epochs before the per-bin
        calibration metrics flatten."""
        return trav_map[:, 0].std(dim=(1, 2)).mean()

    def depth_criterion(self, prediction, target, mask):
        """
        Calculate the depth classification loss.

        Args:
            prediction (torch.Tensor): Tensor containing the predicted depth.
            target (torch.Tensor): Tensor containing the target depth.
            mask (torch.Tensor): Tensor containing the mask.

        Returns:
            torch.Tensor: Depth loss.
        """
        loss = F.cross_entropy(prediction, target, reduction='none') * mask
        return torch.mean(loss)

    @torch.no_grad()
    def _voxelize_for_vis(self, color_img, depth_voxel, intrinsics_full, extrinsics):
        """Rebuild the occupancy grid (B, T, Z, X, Y) for visualization only."""
        B, T = color_img.shape[:2]
        pcloud = self.model.voxelize(
            depth_voxel.view(B * T, *depth_voxel.shape[2:]) * 1e-3,
            intrinsics_full.view(B * T, 3, 3),
            extrinsics.view(B * T, 4, 4))
        return pcloud.view(B, T, *pcloud.shape[1:])

    @torch.no_grad()
    def _draw_bev_path(self, rgb, path):
        """Draw the base-frame path (H, >=2) onto a BEV RGB image (in place).
        Harvested extras past the horizon slots are scattered points, not path."""
        path = path[: self.path_horizon]
        p = path[torch.isfinite(path[:, 0])].cpu().numpy()
        if len(p) == 0:
            return
        u = np.round(self.map_origin[0] - p[:, 0] / self.map_resolution[0]).astype(np.int32)
        v = np.round(self.map_origin[1] - p[:, 1] / self.map_resolution[1]).astype(np.int32)
        pts = np.stack([v, u], -1).reshape(-1, 1, 2)
        cv2.polylines(rgb, [pts], False, PATH_LINE, 1, cv2.LINE_AA)
        cv2.circle(rgb, (int(v[0]), int(u[0])), 2, PATH_START, -1)
        cv2.circle(rgb, (int(v[-1]), int(u[-1])), 2, PATH_END, -1)

    @torch.no_grad()
    def _bev_overlays(self, bev, path, normalize=False):
        """Grayscale BEV maps (N,1,X,Y) -> N RGB uint8 tensors with the path drawn on top.
        The per-sample path (B,H,>=2) is repeated to cover temporal stacks (N = B*T)."""
        bev = bev[:, 0].detach().float().cpu().numpy()
        reps = max(1, bev.shape[0] // path.shape[0])
        overlays = []
        for n in range(bev.shape[0]):
            m = bev[n]
            if normalize:
                m = (m - m.min()) / (m.max() - m.min() + 1e-6)
            else:
                m = np.clip(m, 0, 1)
            rgb = np.repeat((m * 255).astype(np.uint8)[..., None], 3, 2).copy()
            self._draw_bev_path(rgb, path[n // reps])
            overlays.append(torch.from_numpy(rgb))
        return overlays

    @torch.no_grad()
    def _bev_label_overlays(self, trav_map, path, target_trav, trav_weights):
        """Predicted mu BEV colorized red-green, with each labelled path cell drawn
        on top in its label color; gated (zero-weight) cells as gray outlines."""
        mu = trav_map[:, 0].detach().float().cpu().numpy()
        out = []
        for b in range(mu.shape[0]):
            rgb = np.ascontiguousarray(colorize_mu(mu[b])[..., ::-1])
            p = path[b].detach().float().cpu().numpy()
            keep = np.isfinite(p[:, 0])
            t = target_trav[b, keep, 0].detach().float().cpu().numpy()
            w = trav_weights[b, keep, 0].detach().float().cpu().numpy()
            u = np.round(self.map_origin[0] - p[keep, 0] / self.map_resolution[0]).astype(int)
            v = np.round(self.map_origin[1] - p[keep, 1] / self.map_resolution[1]).astype(int)
            colors = colorize_mu(t)[..., ::-1]
            h_map, w_map = rgb.shape[:2]
            for ui, vi, ci, wi in zip(u, v, colors, w):
                if 0 <= ui < h_map and 0 <= vi < w_map:
                    if wi > 0:
                        cv2.circle(rgb, (vi, ui), 2, tuple(int(x) for x in ci), -1)
                    else:
                        cv2.circle(rgb, (vi, ui), 2, (128, 128, 128), 1)
            out.append(rgb)
        return out

    @torch.no_grad()
    def _depth_cam_overlays(self, image, depth_target, depth_mask):
        """Anchor frame with the input depth target blended on top (JET, masked)."""
        B, T = image.shape[:2]
        n_d = int((self.grid_bounds['dbound'][1] - self.grid_bounds['dbound'][0]) / self.grid_bounds['dbound'][2])
        out = []
        for b in range(B):
            idx = (b + 1) * T - 1  # anchor frame in the flattened (B*T) stack
            if self.predict_depth:
                d = depth_target[idx].float().cpu().numpy() / (n_d - 1)
            else:
                d = depth_target[idx].argmax(0).float().cpu().numpy() / (n_d - 1)
            m = depth_mask[idx].float().cpu().numpy() > 0
            rgb = self._frame_rgb(image, b, -1)
            h, w = rgb.shape[:2]
            col = cv2.applyColorMap((np.clip(d, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
            col = cv2.resize(col, (w, h), interpolation=cv2.INTER_NEAREST)
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            blend = rgb.copy()
            blend[m] = (0.55 * col[m] + 0.45 * rgb[m]).astype(np.uint8)
            out.append(blend)
        return out

    @staticmethod
    def _ground_plane(path):
        """Plane z = a*x + b*y + c fitted to the path's ground points near the grid,
        the surface the BEV grid cells are drawn on. Only points within 10 m enter
        the fit (the 30 s path can climb far outside the grid and would tilt it),
        the slope is ridge-regularized against ill-conditioned point layouts, and
        the plane is clamped to the observed z range so extrapolation to the grid
        corners cannot leave the terrain. 0.0 (the base origin) when the path has
        no ground column or no valid points."""
        finite = torch.isfinite(path[:, 0])
        if path.shape[1] <= 5 or not finite.any():
            return 0.0
        p = path[finite].float().cpu().numpy()
        near = np.hypot(p[:, 0], p[:, 1]) <= 10.0
        if near.sum() >= 3:
            p = p[near]
        if len(p) < 3:
            return float(p[0, 5])
        A = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
        AtA = A.T @ A + np.diag([4.0, 4.0, 0.0])
        a, b, c = np.linalg.solve(AtA, A.T @ p[:, 5])
        margin = 0.3
        return (a, b, c, p[:, 5].min() - margin, p[:, 5].max() + margin)

    @torch.no_grad()
    def _project_path_to_cam(self, path, K, extrinsics):
        """Base-frame path (H,>=2) -> anchor-camera pixels (inverse of read_pcloud).
        State column 5 is the ground height under each state; without it the
        points would sit at base height, half a meter above the ground."""
        p = path[torch.isfinite(path[:, 0])]
        z = p[:, 5] if p.shape[1] > 5 else torch.zeros_like(p[:, 0])
        pts = torch.stack([p[:, 0], p[:, 1], z], 1)
        cam = (pts - extrinsics[:3, 3]) @ extrinsics[:3, :3]
        front = cam[:, 2] > 1e-3
        pix = cam @ K.T
        uv = pix[:, :2] / pix[:, 2:3].clamp(min=1e-3)
        return uv.float().cpu().numpy(), front.cpu().numpy()

    @staticmethod
    def _gray_to_rgb(img):
        """(1,H,W) float in [0,1] -> (H,W,3) uint8, for combining with BEV/cam overlays."""
        arr = (img.squeeze(0).clamp(0, 1) * 255).to(torch.uint8)
        return arr.unsqueeze(-1).expand(-1, -1, 3).contiguous()

    @staticmethod
    def _hstack(*overlay_lists, sep=4):
        """Concatenate matching (H,W,3) uint8 tensors from each list side by side."""
        combined = []
        for imgs in zip(*overlay_lists):
            h = max(im.shape[0] for im in imgs)
            padded = [F.pad(im, (0, 0, 0, 0, 0, h - im.shape[0])) if im.shape[0] != h else im for im in imgs]
            separator = torch.zeros(h, sep, 3, dtype=padded[0].dtype)
            row = padded[0]
            for im in padded[1:]:
                row = torch.cat([row, separator, im], dim=1)
            combined.append(row)
        return combined

    @torch.no_grad()
    def _frame_rgb(self, image, b, t):
        """(B, T, 3, H, W) float [0,1] -> one (H, W, 3) uint8 frame."""
        img = image[b, t].detach().float().cpu().clamp(0, 1).numpy()
        return (img.transpose(1, 2, 0) * 255).astype(np.uint8).copy()

    @torch.no_grad()
    def _draw_path_on_frame(self, img, path, K, extrinsics):
        """Project the base-frame path into one frame and draw it (in place)."""
        uv, front = self._project_path_to_cam(path, K, extrinsics)
        h, w = img.shape[:2]
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.flatnonzero(inside)
        if len(idx):
            pts = uv.round().astype(np.int32)
            # split at gaps so exits/re-entries are not bridged with straight lines
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                cv2.polylines(img, [pts[run].reshape(-1, 1, 2)], False, PATH_LINE, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[idx[0]]), 4, PATH_START, -1)
            cv2.circle(img, tuple(pts[idx[-1]]), 4, PATH_END, -1)
        return img

    @torch.no_grad()
    def _cam_overlays(self, image, path, K, extrinsics):
        """Anchor RGB frame per batch element with the future path projected on top."""
        overlays = []
        for b in range(image.shape[0]):
            img = self._frame_rgb(image, b, -1)
            self._draw_path_on_frame(img, path[b], K[b, -1], extrinsics[b, -1])
            overlays.append(torch.from_numpy(img))
        return overlays

    @staticmethod
    def _stride_batch(n, cap):
        """Evenly-spaced indices into [0, n) capped at `cap`."""
        return np.linspace(0, n - 1, min(cap, n)).round().astype(int)

    def visualize(self, image, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, path, intrinsics, extrinsics, target_trav=None, trav_weights=None, mu_logits=None, depth_voxel=None, intrinsics_full=None, prefix='train'):
        """
        Visualize the training/validation results.

        Args:
            image (torch.Tensor): Tensor containing the images.
            pcloud (torch.Tensor): Tensor containing the point clouds.
            trav_map (torch.Tensor): Tensor containing the traversability map.
            pred_depth (torch.Tensor): Tensor containing the predicted depth.
            depth_target (torch.Tensor): Tensor containing the target depth.
            depth_mask (torch.Tensor): Tensor containing the depth mask.
            debug (torch.Tensor): Tensor containing the debug information.
            path (torch.Tensor): Tensor containing the path states (B, H, >=2).
            intrinsics (torch.Tensor): Input-size camera K (B, T, 3, 3).
            extrinsics (torch.Tensor): Cam-to-base per frame (B, T, 4, 4).
            target_trav (torch.Tensor): Label (mu, nu) along the path (B, H, 2).
            trav_weights (torch.Tensor): Loss weights along the path (B, H, 2).
            mu_logits (torch.Tensor): Auxiliary mu head output (see TravNet.forward).
            depth_voxel (torch.Tensor): Full-res depth stack (B, T, H, W) in int16 mm.
            intrinsics_full (torch.Tensor): Full-res camera K matching depth_voxel.
            prefix (str): Prefix for the log keys.
        """
        # Batch elements are 1-frame-apart sliding windows (near-duplicate scenes),
        # so stride through the batch dim to spread the logged panel across it
        # instead of showing self.vis_batch_size consecutive near-identical frames.
        B_full, T = image.shape[:2]
        b_idx = self._stride_batch(B_full, self.vis_batch_size)
        flat_idx = (b_idx[:, None] * T + np.arange(T)[None, :]).ravel()

        image, pcloud, trav_map, path, intrinsics, extrinsics = (
            t[b_idx] for t in (image, pcloud, trav_map, path, intrinsics, extrinsics))
        if target_trav is not None:
            target_trav = target_trav[b_idx]
        if trav_weights is not None:
            trav_weights = trav_weights[b_idx]
        if mu_logits is not None:
            mu_logits = mu_logits[b_idx]
        if depth_voxel is not None:
            depth_voxel = depth_voxel[b_idx]
        if intrinsics_full is not None:
            intrinsics_full = intrinsics_full[b_idx]
        pred_depth, depth_target, depth_mask, debug = (
            t[flat_idx] for t in (pred_depth, depth_target, depth_mask, debug))

        # Camera input (anchor frame) with the future path projected on top
        self.logger.log_image(
            key=prefix + "/images",
            images=[im.numpy() for im in self._cam_overlays(image, path, intrinsics, extrinsics)],
        )

        # Predicted mu splatted onto the anchor frame (red = slippery, green = grippy)
        mu = trav_map[:, 0].detach().float().cpu().numpy()
        mu_cam = []
        for b in range(image.shape[0]):
            img = image[b, -1].detach().float().cpu().clamp(0, 1).numpy()
            bgr = np.ascontiguousarray((img.transpose(1, 2, 0)[..., ::-1] * 255).astype(np.uint8))
            K = intrinsics[b, -1].float().cpu().numpy()
            ext = extrinsics[b, -1].float().cpu().numpy()
            depth_m = depth_voxel[b, -1].float().cpu().numpy() * 1e-3 if depth_voxel is not None else None
            K_full = intrinsics_full[b, -1].float().cpu().numpy() if intrinsics_full is not None else None
            mu_cam.append(np.ascontiguousarray(overlay_bev_on_image(
                bgr, mu[b], K, ext, self.grid_bounds,
                ground_z=self._ground_plane(path[b]),
                depth=depth_m, depth_K=K_full)[..., ::-1]))
        self.logger.log_image(key=prefix + "/mu_cam", images=mu_cam)

        B, T = image.shape[:2]

        if self.debug_vis_labels and target_trav is not None:
            # Supervision as the loss sees it: label mu (red-green) at weighted path
            # cells, gray outlines where the visibility/observability gates zeroed it
            label_cam = []
            for b in range(B):
                bgr = np.ascontiguousarray(self._frame_rgb(image, b, -1)[..., ::-1])
                out = overlay_path_on_image(
                    bgr, path[b].detach().float().cpu(), target_trav[b].detach().float().cpu(),
                    trav_weights[b].detach().float().cpu(), intrinsics[b, -1].float().cpu().numpy(),
                    extrinsics[b, -1].float().cpu().numpy(), show_gated=True)
                label_cam.append(np.ascontiguousarray(out[..., ::-1]))
            self.logger.log_image(key=prefix + "/label_cam", images=label_cam)
            # Predicted mu BEV in color with the label cells drawn on top:
            # prediction vs supervision in one map
            self.logger.log_image(
                key=prefix + "/bev_label",
                images=self._bev_label_overlays(trav_map, path, target_trav, trav_weights))

        if self.debug_vis_geom:
            # Path reprojected into every input frame with that frame's own
            # extrinsics: ego-motion or extrinsics errors show as the path
            # detaching from the ground in the older frames
            frame_cols = []
            for t in range(T):
                col = []
                for b in range(B):
                    img = self._frame_rgb(image, b, t)
                    self._draw_path_on_frame(img, path[b], intrinsics[b, t], extrinsics[b, t])
                    col.append(torch.from_numpy(img))
                frame_cols.append(col)
            self.logger.log_image(
                key=prefix + "/geom_frames",
                images=[im.numpy() for im in self._hstack(*frame_cols)])

        if self.debug_vis_depth:
            # Input depth (JET colormap) blended over the anchor frame: checks
            # depth-RGB alignment, the border crop, and the K scaling
            self.logger.log_image(
                key=prefix + "/depth_cam",
                images=self._depth_cam_overlays(image, depth_target, depth_mask))

        # BEV inputs: point-cloud occupancy density | lift-splat feature density (both sanity checks)
        pcloud = torch.mean(pcloud, dim=2).view(-1, 1, *pcloud.shape[3:])
        pcloud_overlays = self._bev_overlays(pcloud, path, normalize=True)
        debug_overlays = self._bev_overlays(torch.sum(debug, dim=1, keepdim=True), path, normalize=True)
        self.logger.log_image(key=prefix + "/bev_inputs", images=[im.numpy() for im in self._hstack(pcloud_overlays, debug_overlays)])

        # BEV outputs: predicted mu | nu (| epistemic uncertainty when evidential)
        columns = [self._bev_overlays(trav_map[:, :1], path), self._bev_overlays(trav_map[:, 1:], path)]
        if self.evidential and mu_logits is not None:
            # NIG variance of the mean, beta / (v * (alpha - 1)); per-image normalized
            v, alpha, beta = mu_logits[:, 0:1], mu_logits[:, 1:2], mu_logits[:, 2:3]
            unc = beta / (v * (alpha - 1).clamp_min(self.eps))
            columns.append(self._bev_overlays(unc, path, normalize=True))
        self.logger.log_image(key=prefix + "/bev_outputs", images=[im.numpy() for im in self._hstack(*columns)])

        # Depth (camera frame, no BEV path): predicted | target | validity mask
        n_d = (self.grid_bounds['dbound'][1] - self.grid_bounds['dbound'][0])/self.grid_bounds['dbound'][2]
        depth_pred = torch.argmax(pred_depth, dim=1, keepdim=True)/(n_d-1)

        if self.predict_depth:
            depth_target = depth_target.unsqueeze(1)/(n_d-1)
        else:
            depth_target = depth_target.argmax(1).unsqueeze(1)/(n_d-1)

        pred_overlays = [self._gray_to_rgb(t) for t in depth_pred.float().cpu()]
        target_overlays = [self._gray_to_rgb(t) for t in depth_target.float().cpu()]
        mask_overlays = [self._gray_to_rgb(t) for t in depth_mask.unsqueeze(1).float().cpu()]
        self.logger.log_image(key=prefix + "/depth", images=[im.numpy() for im in self._hstack(pred_overlays, target_overlays, mask_overlays)])