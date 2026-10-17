# Taken from wayfaster repository (https://github.com/matval/wayfaster)
# Modifications: fixed import of TravNet for local package layout.
import os
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .models.traversability_net import TravNet
from .. import adaptations


class TrainingModule(adaptations.VisualizationMixin, pl.LightningModule):
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
        # Harvested low-mu extras sit past the horizon: no temporal discount
        self.path_horizon = configs.TRAINING.HORIZON
        harvest_extra = getattr(configs.DATASET, "HARVEST_LOWMU_EXTRA", 0)
        if harvest_extra:
            gamma = torch.cat([gamma, torch.ones(harvest_extra)])
        self.gamma = nn.Parameter(gamma, requires_grad=False)
        # Mu loss selector. Unset (None) means plain regression.
        mu_bins = getattr(configs.MODEL, "MU_BINS", 0)
        bmc_init = getattr(configs.TRAINING, "BMC_SIGMA_INIT", 0.0)
        loss_type = configs.TRAINING.LOSS_TYPE or "regression"
        valid = {"regression", "bmc", "binned_ce", "class3", "evidential", "rizz", "focal_r"}
        assert loss_type in valid, f"LOSS_TYPE must be one of {valid}, got {loss_type!r}"
        if loss_type == "bmc":
            assert bmc_init > 0, "LOSS_TYPE=bmc needs TRAINING.BMC_SIGMA_INIT > 0"
        else:
            bmc_init = 0.0
        if loss_type in ("binned_ce", "class3"):
            assert mu_bins > 0, f"LOSS_TYPE={loss_type} needs MODEL.MU_BINS > 0"
        elif not getattr(configs.MODEL, "MU_PIXEL_LINEAR", False):
            # ADAPTATION: pixel-linear head emits bins regardless of loss; mu is their expectation
            mu_bins = 0
        evidential = loss_type == "evidential"
        # class3: thresholded mu classes, logit-adjusted CE (Menon et al. 2021), priors added in the loss only
        self.mu_class_loss = loss_type == "class3"
        if self.mu_class_loss:
            thresholds = list(configs.TRAINING.MU_CLASS_THRESHOLDS)
            priors = list(configs.TRAINING.MU_CLASS_PRIORS)
            assert mu_bins == len(thresholds) + 1 == len(priors)
            self.mu_class_thresholds = nn.Parameter(torch.tensor(thresholds), requires_grad=False)
            self.mu_class_log_priors = nn.Parameter(torch.tensor(priors).log(), requires_grad=False)
        # Balanced MSE (BMC) noise scale, learnable; None = smooth-L1 mu term
        self.bmc_sigma = nn.Parameter(torch.tensor(float(bmc_init))) if bmc_init > 0 else None
        self.learning_rate = configs.OPTIMIZER.LR
        self.lr_ref_batchsize = configs.OPTIMIZER.LR_REF_BATCHSIZE
        self.backbone_lr_factor = getattr(configs.OPTIMIZER, "BACKBONE_LR_FACTOR", 1.0)
        self.batchsize = configs.TRAINING.BATCHSIZE
        self.weight_decay = configs.OPTIMIZER.WEIGHT_DECAY
        self.depth_weight = configs.TRAINING.DEPTH_WEIGHT
        self.overpred_weight = getattr(configs.TRAINING, "TRAV_OVERPRED_WEIGHT", 1.0)
        self.smooth_l1_beta = getattr(configs.TRAINING, "SMOOTH_L1_BETA", 0.1)
        # ADAPTATION: trav_loss normalization knob ("weighted" adapted, "mean" original)
        self.loss_norm = getattr(configs.TRAINING, "LOSS_NORM", "weighted")
        # ADAPTATION: train-only mu distribution matching (grandtour histogram)
        _mmk = getattr(configs.DATASET, "MU_MATCH_KEEP", None)
        self.mu_match_keep = torch.tensor(_mmk, dtype=torch.float32) if _mmk else None
        self.mu_head_bins = mu_bins
        # Bins the loss is taken over, as opposed to the ones the head emits: only
        # binned_ce replaces the mu term with a cross-entropy.
        self.mu_ce_bins = mu_bins if loss_type == "binned_ce" else 0
        # ADAPTATION: Gaussian instead of two-hot target for the binned head (0 = two-hot)
        self.mu_bin_sigma = getattr(configs.TRAINING, "MU_BIN_SIGMA", 0.0)
        # ADAPTATION: Focal-R weighting of the mu regression term
        self.focal_r = loss_type == "focal_r"
        self.focal_beta = getattr(configs.TRAINING, "FOCAL_BETA", 10.0)
        self.focal_gamma = getattr(configs.TRAINING, "FOCAL_GAMMA", 2.0)
        # Evidential (NIG) mu head, Deep Evidential Regression loss (Amini et al. 2020)
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
            freeze_backbone=getattr(configs.MODEL, "FREEZE_BACKBONE", False),
            backbone=getattr(configs.MODEL, "BACKBONE", "resnet34"),
            dino_feat_dim=getattr(configs.MODEL, "DINO_FEAT_DIM", 768),
            dino_neck_dim=getattr(configs.MODEL, "DINO_NECK_DIM", 256),
            dino_noise=getattr(configs.MODEL, "DINO_FEAT_NOISE", 0.0),
            mu_pixel_linear=getattr(configs.MODEL, "MU_PIXEL_LINEAR", False),
            head_drop=getattr(configs.MODEL, "HEAD_DROPOUT", 0.0),
            slim_decoder=getattr(configs.MODEL, "SLIM_DECODER", False),
            wrizz_decoder=getattr(configs.MODEL, "WRIZZ_DECODER", False),
        )

        # Convert to channels_last to avoid DDP bucket stride mismatch warnings
        # when torch.compile enables cuDNN channels-last optimizations.
        # We only apply this to specific submodules to avoid RuntimeError on 3D convolutions.
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                m.to(memory_format=torch.channels_last)
            elif isinstance(m, (nn.Conv3d, nn.BatchNorm3d)):
                m.to(memory_format=torch.channels_last_3d)

        # ADAPTATION: mean teacher (adaptations/mean_teacher.py), copied before compile to stay eager
        self.mt_weight = getattr(configs.TRAINING, "MEAN_TEACHER_WEIGHT", 0.0)
        self.mean_teacher = None
        if self.mt_weight > 0:
            self.mean_teacher = adaptations.MeanTeacher(
                self.model,
                decay=configs.TRAINING.MEAN_TEACHER_DECAY,
                ramp_steps=configs.TRAINING.MEAN_TEACHER_RAMP,
                jitter=tuple(configs.TRAINING.MEAN_TEACHER_JITTER),
            )
        self._student_core = self.model

        if configs.TRAINING.COMPILE:
            import torch._dynamo as _dynamo
            # BasicBlock is reused with many shapes and train/val modes
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
        # ADAPTATION: panels named here are never logged (wandb image build-up)
        self.vis_skip = set(getattr(configs.TRAINING, "VIS_PANELS_SKIP", []))
        self.debug_vis_labels = getattr(configs.TRAINING, "DEBUG_VIS_LABELS", False)
        self.debug_vis_geom = getattr(configs.TRAINING, "DEBUG_VIS_GEOM", False)
        self.debug_vis_depth = getattr(configs.TRAINING, "DEBUG_VIS_DEPTH", False)

        # ADAPTATION: observation-only mu calibration metrics, per validation mission
        # (adaptations/metrics.py, adaptations/mission_ids.py)
        self.mission_names = list(getattr(configs.DATASET, "VALID_MISSIONS", []))
        self.mu_metrics = adaptations.MuBinMetrics(mission_names=self.mission_names)

        # ADAPTATION: W-RIZZ ordinal mu supervision (adaptations/rizz.py). HDR is logged for every run.
        self.rizz = loss_type == "rizz"
        self.rizz_margin = configs.TRAINING.RIZZ_MARGIN
        self.rizz_weight = configs.TRAINING.RIZZ_WEIGHT
        self.rizz_pairs = configs.TRAINING.RIZZ_PAIRS
        self.rizz_kw = dict(
            deadband=configs.TRAINING.RIZZ_DEADBAND,
            tie_band=configs.TRAINING.RIZZ_TIE_BAND,
            cross_frac=configs.TRAINING.RIZZ_CROSS_FRACTION,
            min_sep=configs.TRAINING.RIZZ_MIN_SEP,
            ineq_frac=configs.TRAINING.RIZZ_INEQ_FRACTION,
            ineq_intra_only=configs.TRAINING.RIZZ_INEQ_INTRA_ONLY,
        )
        # eval pairs come from rizz.EVAL_SPECS, not rizz_kw, to stay comparable across runs
        self.hdr_metrics = adaptations.RizzMetrics(configs.TRAINING.RIZZ_HDR_PAIRS)

        # ADAPTATION: human W-RIZZ pair term on top of LOSS_TYPE (adaptations/human_pairs.py)
        self.trav_weight = getattr(configs.TRAINING, "TRAV_WEIGHT", 1.0)
        self.human_weight = configs.TRAINING.HUMAN_WEIGHT
        self.human_margin = configs.TRAINING.HUMAN_MARGIN
        self.human_point_pairs = configs.TRAINING.HUMAN_POINT_PAIRS
        self.human_point_mass = configs.TRAINING.HUMAN_POINT_MASS
        self.human_point_samples = configs.TRAINING.HUMAN_POINT_SAMPLES
        self.human_metrics = None
        if self.human_weight > 0:
            assert configs.TRAINING.HUMAN_TARGET == "bev", (
                "only HUMAN_TARGET='bev' is implemented: image-space supervision needs "
                "TravNet to return its per-pixel mu tensor (traversability_net.py:316)")
            table = adaptations.PairTable(configs.DATASET.HUMAN_INDEX_ROOT)
            self.human_pairs = len(table)
            self.human_metrics = adaptations.HumanHdrMetrics(table)
            # ADAPTATION: HDR over the trained pairs; the train/valid gap shows overfitting
            self.human_train_metrics = adaptations.HumanHdrMetrics(table)

        # ADAPTATION: which optional trailing batch elements this run's samples carry
        self.extra_flags = adaptations.mission_ids.flags(configs)
        self.uses_dino = getattr(configs.MODEL, "BACKBONE", "resnet34") == "dino"

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

        # DINO features are precomputed separately; fail early if the cache is missing or stale
        if getattr(configs.MODEL, "BACKBONE", "resnet34") == "dino":
            cache_root = pathlib.Path(configs.DATASET.DINO_CACHE_ROOT)
            for mission in missions:
                meta_path = cache_root / mission / "meta.npz"
                assert meta_path.exists(), (
                    f"DINO feature cache missing for {mission}. Run:\n"
                    f"  uv run python -m grandtour_extraction.cache_dino_features --missions {mission}"
                )
                meta = np.load(meta_path, allow_pickle=True)
                assert str(meta["model"]) == configs.MODEL.DINO_MODEL, (
                    f"{mission}: cached model {meta['model']} != configured {configs.MODEL.DINO_MODEL}"
                )
                assert list(meta["input_size"]) == list(configs.MODEL.DINO_INPUT_SIZE), (
                    f"{mission}: cached input_size {meta['input_size']} != configured {configs.MODEL.DINO_INPUT_SIZE}"
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
        # Get data (trailing elements = cached DINO features, mission index, human pairs)
        color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics, path, target_trav, trav_weights, depth_target, depth_mask, *extra = batch
        image_feats, _, human = adaptations.mission_ids.split(extra, *self.extra_flags)

        # Pointcloud dropout: zero whole depth windows so the GPU builds an empty grid
        depth_raw = depth_voxel  # undropped, for the viz panels
        if self.pcloud_dropout > 0:
            drop = torch.rand(depth_voxel.shape[0], device=depth_voxel.device) < self.pcloud_dropout
            if drop.any():
                depth_voxel = depth_voxel.clone()
                depth_voxel[drop] = 0

        # Forward pass
        trav_map, mu_logits, pred_depth, debug = self.model(color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics, depth_target, image_feats=image_feats)

        # Calculate traversability loss
        with torch.set_grad_enabled(bool(self.trav_weight)):
            trav_loss, _ = self.trav_criterion(path, trav_map, target_trav, trav_weights, mu_logits)

        # ADAPTATION: update graded low-mu metrics on train batches (logged at validation)
        self.mu_metrics.update_train(self, path, trav_map, target_trav, trav_weights)
        self.hdr_metrics.update(self, path, trav_map, target_trav, trav_weights, batch_idx, train=True)

        # Calculate depth classification loss
        depth_target = depth_target.view(-1, *depth_target.shape[2:])
        depth_mask = depth_mask.view(-1, *depth_mask.shape[2:])
        depth_loss = self.depth_criterion(pred_depth, depth_target, depth_mask)

        # ADAPTATION: TRAV_WEIGHT 0 keeps the RHE loss as a logged diagnostic only
        weighted_trav = self.trav_weight * trav_loss if self.trav_weight else 0.0
        if self.train_depth:
            loss = weighted_trav + self.depth_weight*depth_loss
        else:
            loss = weighted_trav

        # ADAPTATION: human W-RIZZ ordinal term, summed on top of the mu loss
        if self.human_weight > 0:
            human_loss, n_human, n_half = adaptations.human_loss(
                trav_map, human, self.grid_bounds, self.human_margin,
                self.human_pairs, self.human_point_pairs, self.human_point_mass,
                self.human_point_samples)
            loss = loss + self.human_weight * human_loss
            self.human_train_metrics.update(trav_map, human, self.grid_bounds)
            self.log("human/train_loss", human_loss, on_step=True, on_epoch=False)
            self.log("human/train_batch_pairs", float(n_human), on_step=True, on_epoch=False)
            # half pairs mean the sampler no longer keeps both windows together
            self.log("human/train_half_pairs", float(n_half), on_step=True, on_epoch=False)

        # ADAPTATION: mean teacher consistency over every BEV cell, not just the
        # annotated ones (see adaptations/mean_teacher.py)
        if self.mean_teacher is not None:
            teacher_map = self.mean_teacher.predict(
                color_img, depth_raw, intrinsics_full, inv_intrinsics, extrinsics,
                depth_target, image_feats=image_feats)
            mt = adaptations.mt_consistency(trav_map, teacher_map)
            w = self.mt_weight * self.mean_teacher.weight(self.global_step)
            loss = loss + w * mt
            self.log("train/mt_consistency", mt, on_step=True, on_epoch=False)
            self.log("train/mt_weight", w, on_step=True, on_epoch=False)

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
        self.log("train/depth_mae", adaptations.depth_mae(self.grid_bounds, pred_depth, depth_target, depth_mask), on_step=True, on_epoch=False)
        self.log("train/mu_spread", adaptations.mu_spread(trav_map), on_step=True, on_epoch=False)

        return loss

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.mean_teacher is not None:
            self.mean_teacher.update(self._student_core)
        if hasattr(self, '_batch_start_time'):
            step_time = time.time() - self._batch_start_time
            self.log("perf/time_per_step", step_time, on_step=True, on_epoch=False)

    def on_train_epoch_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        if hasattr(self, '_epoch_start_time'):
            epoch_time = time.time() - self._epoch_start_time
            self.log("perf/time_per_epoch", epoch_time, on_step=False, on_epoch=True, sync_dist=True)
        # train confusion/calibration normally logs from the validation hook, which
        # without a valid loader never fires
        if not self.trainer.val_dataloaders:
            self.mu_metrics.log_train(self)
            self.hdr_metrics.log(self, "train", train=True)

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): Batch of data containing images, point clouds, intrinsics, extrinsics, paths, targets, and masks.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        # Get data (trailing elements = cached DINO features, mission index, human pairs)
        color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics, path, target_trav, trav_weights, depth_target, depth_mask, *extra = batch
        image_feats, mission_id, human = adaptations.mission_ids.split(extra, *self.extra_flags)

        # Forward pass
        trav_map, mu_logits, pred_depth, debug = self.model(color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics, depth_target, image_feats=image_feats)

        # Calculate traversability loss
        trav_loss, trav_error = self.trav_criterion(path, trav_map, target_trav, trav_weights, mu_logits)

        # ADAPTATION: update graded low-mu metrics (per-0.1-bin MAE + calibration)
        self.mu_metrics.update_valid(self, path, trav_map, target_trav, trav_weights, mission_id)
        self.hdr_metrics.update(self, path, trav_map, target_trav, trav_weights, batch_idx, train=False)
        if self.human_metrics is not None:
            self.human_metrics.update(trav_map, human, self.grid_bounds)

        # Calculate depth classification loss
        depth_target = depth_target.view(-1, *depth_target.shape[2:])
        depth_mask = depth_mask.view(-1, *depth_mask.shape[2:])
        depth_loss = self.depth_criterion(pred_depth, depth_target, depth_mask)

        weighted_trav = self.trav_weight * trav_loss if self.trav_weight else 0.0
        if self.train_depth:
            loss = weighted_trav + self.depth_weight*depth_loss
        else:
            loss = weighted_trav

        # Visualize a fixed set of batches spread across the whole shard
        if batch_idx in self._val_vis_batches:
            pcloud = self._voxelize_for_vis(color_img, depth_voxel, intrinsics_full, extrinsics)
            self.visualize(color_img, pcloud, trav_map, pred_depth, depth_target, depth_mask, debug, path, inv_intrinsics, extrinsics, target_trav, trav_weights, mu_logits, depth_voxel=depth_voxel, intrinsics_full=intrinsics_full, prefix='valid')

        self.log("valid/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/trav_loss", trav_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/trav_error", trav_error, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/depth_loss", depth_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/depth_mae", adaptations.depth_mae(self.grid_bounds, pred_depth, depth_target, depth_mask), on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/mu_spread", adaptations.mu_spread(trav_map), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_start(self):
        # VIS_INTERVAL can exceed the batch count, so spread a few panels across the shard
        n = self.trainer.num_val_batches[0]
        if isinstance(n, (int, float)) and np.isfinite(n) and n > 0:
            self._val_vis_batches = {int(i) for i in np.linspace(0, int(n) - 1, min(4, int(n)))}
        else:
            self._val_vis_batches = {0}
        self.mu_metrics.reset_confusion()

    def on_validation_epoch_end(self):
        # ADAPTATION: per-bin calibration + confusion image (see adaptations/metrics.py)
        self.mu_metrics.on_validation_epoch_end(self)
        # ADAPTATION: ordinal HDR on the same pairs for train and valid
        self.hdr_metrics.log(self, "train", train=True)
        self.hdr_metrics.log(self, "valid")
        # ADAPTATION: HDR over the human pairs, both splits (adaptations/human_pairs.py)
        if self.human_metrics is not None:
            self.human_metrics.log(self, "valid")
            self.human_train_metrics.log(self, "train")

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
            if ("encoder.backbone" in n or "encoder.layer" in n) and p.requires_grad
        ]
        pretrained_ids = {id(p) for p in pretrained}
        rest = [
            p for p in self.model.parameters()
            if id(p) not in pretrained_ids and p.requires_grad
        ]
        groups = [{"params": rest, "lr": lr}]
        if pretrained:
            groups.append({"params": pretrained, "lr": lr * self.backbone_lr_factor})
        if self.bmc_sigma is not None:
            # single scalar; the BMC reference implementation trains it at ~1e-2
            groups.append({"params": [self.bmc_sigma], "lr": 1e-2, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)

        # ADAPTATION: CosineSchedule keeps the configured T_max across a resume
        scheduler = adaptations.CosineSchedule(optimizer, T_max=self.epochs, eta_min=1e-5 * lr_scale)
        
        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        # bf16-mixed has no GradScaler: skip updates with non-finite gradients.
        # Fused norms + one .item() avoids a GPU sync per parameter.
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

        # ADAPTATION: reshape train mu supervision to a target histogram (grandtour)
        if self.training and self.mu_match_keep is not None:
            mask = mask & adaptations.mu_match_mask(target_trav[..., 0], self.mu_match_keep.to(mask.device))

        # Calculate loss
        gamma = self.gamma[:traversability.shape[1]].view(1,-1,1).expand(traversability.shape[0],-1,2)
        error = F.smooth_l1_loss(traversability, target_trav, reduction='none', beta=self.smooth_l1_beta)
        # ADAPTATION: alternative mu parameterizations swap the mu error column;
        # nu keeps its smooth-L1 term.
        train_error = error
        if self.mu_class_loss:
            logits, _ = self._sample_path(states, mu_logits)
            ce = adaptations.class3_ce(logits, target_trav[..., 0], self.mu_class_thresholds, self.mu_class_log_priors)
            train_error = torch.stack([ce, error[..., 1]], dim=-1)
        elif self.mu_ce_bins:
            logits, _ = self._sample_path(states, mu_logits)
            ce = adaptations.binned_ce(logits, target_trav[..., 0], self.mu_ce_bins, self.mu_bin_sigma)
            train_error = torch.stack([ce, error[..., 1]], dim=-1)
        elif self.focal_r:
            residual = traversability[..., 0] - target_trav[..., 0]
            mu_error = adaptations.focal_r(error[..., 0], residual, self.focal_beta, self.focal_gamma)
            train_error = torch.stack([mu_error, error[..., 1]], dim=-1)
        elif self.evidential:
            aux, _ = self._sample_path(states, mu_logits)
            nig = adaptations.evidential_loss(traversability[..., 0], target_trav[..., 0], aux, self.evidential_lambda)
            train_error = torch.stack([nig, error[..., 1]], dim=-1)
        # ADAPTATION: asymmetric penalty on optimistic (over-)predictions
        train_error = adaptations.overpred_weight(train_error, traversability, target_trav, self.overpred_weight)
        weight = target_weights * gamma
        if self.rizz:
            # ADAPTATION: W-RIZZ pairs replace the mu term (pair sampling balances, no LDS weight)
            mu_loss = self.rizz_weight * adaptations.pair_loss(
                traversability[..., 0], target_trav[..., 0], states[..., :2],
                mask & (weight[..., 0] > 0), self.rizz_margin,
                n_pairs=self.rizz_pairs, **self.rizz_kw)
            nu_loss = adaptations.reduce_trav_loss(train_error[..., 1], weight[..., 1], mask, self.loss_norm, self.eps)
            loss = mu_loss + nu_loss
        elif self.bmc_sigma is not None:
            # ADAPTATION: balanced MSE replaces the mu term; nu keeps its weighted smooth-L1
            mu_loss = adaptations.bmc_loss(
                traversability[..., 0], target_trav[..., 0], weight[..., 0] * mask,
                self.bmc_sigma, self.eps)
            nu_loss = adaptations.reduce_trav_loss(train_error[..., 1], weight[..., 1], mask, "weighted", self.eps)
            loss = mu_loss + nu_loss
        else:
            # ADAPTATION: LOSS_NORM selects weighted-average (adapted) vs plain mean (original)
            loss = adaptations.reduce_trav_loss(train_error, weight, mask, self.loss_norm, self.eps)

        # Error stays smooth-L1 on (mu, nu) in both modes, comparable across runs
        error = torch.mean(error[mask,:])

        return loss, error

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
