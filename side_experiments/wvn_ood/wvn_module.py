"""WVN-style OOD self-supervision module for the traversability network."""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from traversability_network import adaptations
from traversability_network.network.trainer import TrainingModule


class LightDecoder(nn.Module):
    """Local-RF BEV decoder: n plain 3x3 conv blocks (GN + dropout), ~0.15M params
    vs the 3.1M from-scratch ResNet18. Receptive field ~1 m instead of the whole
    map, so a cell's mu cannot condition on scene layout."""

    def __init__(self, C, n_blocks=4, p_drop=0.2):
        super().__init__()
        layers = []
        for _ in range(n_blocks):
            layers += [
                nn.Conv2d(C, C, 3, padding=1, bias=False),
                nn.GroupNorm(8, C),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p_drop),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LocalTemporalBlock(nn.Module):
    """TemporalBlock without the pyramid pooling (a full-map spatial mean broadcast
    to every cell is a scene-fingerprint channel, i.e. memorization fuel) and with
    GN instead of BN (batch 8 of correlated windows is BN's worst case)."""

    def __init__(self, C):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(C, C // 2, kernel_size=1, bias=False),
            nn.GroupNorm(8, C // 2), nn.ReLU(inplace=True),
            nn.ConstantPad3d(padding=(1, 1, 1, 1, 1, 0), value=0),
            nn.Conv3d(C // 2, C // 2, kernel_size=(2, 3, 3), bias=False),
            nn.GroupNorm(8, C // 2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(C, C // 2, kernel_size=1, bias=False),
            nn.GroupNorm(8, C // 2), nn.ReLU(inplace=True))
        self.aggregation = nn.Sequential(
            nn.Conv3d(C, C, kernel_size=1, bias=False),
            nn.GroupNorm(8, C), nn.ReLU(inplace=True))

    def forward(self, x):
        return x + self.aggregation(torch.cat([self.conv1(x), self.conv2(x)], dim=1))


class LocalTemporalModel(nn.Module):
    def __init__(self, C, temporal_length):
        super().__init__()
        self.model = nn.Sequential(*[LocalTemporalBlock(C) for _ in range(temporal_length - 1)])

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.model(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x[:, -1, None]


class WvnTrainingModule(TrainingModule):
    def __init__(self, configs):
        light = getattr(configs.WVN, "LIGHT_BEV", False)
        compile_flag = configs.TRAINING.COMPILE
        if light:
            # build uncompiled, swap the BEV stack, then compile the final graph
            configs = copy.deepcopy(configs)
            configs.TRAINING.COMPILE = False
        super().__init__(configs)
        if light:
            m = self.model
            pcloud_dim = m.int_nx[2] if m.fuse_pcloud else 0
            m.bev_compressor = nn.Sequential(
                nn.Conv2d(m.camC * m.int_nx[2] + pcloud_dim, m.camC, 3, padding=1, bias=False),
                nn.GroupNorm(8, m.camC), nn.ReLU(inplace=True))
            m.temporal_model = LocalTemporalModel(m.camC, configs.MODEL.TIME_LENGTH)
            m.decoder = LightDecoder(m.latent_dim)
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    mod.to(memory_format=torch.channels_last)
                elif isinstance(mod, nn.Conv3d):
                    mod.to(memory_format=torch.channels_last_3d)
            if compile_flag:
                import torch._dynamo as _dynamo
                _dynamo.config.recompile_limit = 64
                self.model = torch.compile(self.model)
        g = np.load(configs.WVN.GATE_PATH)
        # non-persistent: checkpoints stay loadable as plain TrainingModule runs
        self.register_buffer(
            "wvn_mean", torch.from_numpy(g["mean"]).float().view(1, -1, 1, 1), persistent=False)
        self.register_buffer(
            "wvn_prec", torch.from_numpy(g["precision"]).float()[..., None, None], persistent=False)
        self.wvn_weight = configs.WVN.WEIGHT
        self.wvn_tau_q = configs.WVN.TAU_QUANTILE
        self.wvn_min_cells = configs.WVN.MIN_CELLS
        self.wvn_vis_min = configs.WVN.VIS_MIN
        self.wvn_max_dist = configs.WVN.MAX_DIST
        self.wvn_z_max = configs.WVN.Z_MAX
        zb = configs.MODEL.GRID_BOUNDS["zbound"]
        self.register_buffer(
            "wvn_z_vals", torch.arange(zb[0], zb[1], zb[2]).flip(0), persistent=False)

    @torch.no_grad()
    def _wvn_gate(self, image_feats, intrinsics, extrinsics, depth_target, depth_mask, path):
        """Computes the BEV OOD gate mask, score map, and splat density for a batch."""
        B, T = image_feats.shape[:2]
        dev = image_feats.device
        with torch.autocast(device_type=dev.type, enabled=False):
            f = image_feats.view(B * T, *image_feats.shape[2:]).float()
            d = F.normalize(f, p=2, dim=1) - self.wvn_mean
            s = (d * F.conv2d(d, self.wvn_prec)).sum(1, keepdim=True)  # (B*T, 1, h, w)
            dt = depth_target.view(B * T, *depth_target.shape[2:]).long()
            dm = depth_mask.view(B * T, *depth_mask.shape[2:]).float()
            D = self.model.D
            dl = F.one_hot(dt.clamp(0, D - 1), D).permute(0, 3, 1, 2).float() * dm.unsqueeze(1)
            s = F.interpolate(s, size=dl.shape[-2:], mode="bilinear", align_corners=False)
            x = dl.unsqueeze(1) * torch.cat([s, torch.ones_like(s)], 1).unsqueeze(2)
            geom = self.model.get_inv_geometry(intrinsics.float(), extrinsics.float())
            bev = self.model.sample2bev(geom, x)  # (B, T, 2, Z, X, Y)
            # only near-ground z-levels: the full column mixes canopy above the
            # trail into ground-cell scores, washing out the obstacle contrast
            zsel = self.wvn_z_vals <= self.wvn_z_max
            num = bev[:, :, 0, zsel].sum(dim=(1, 2))
            den = bev[:, :, 1, zsel].sum(dim=(1, 2))
            score_bev = num / (den + self.eps)

            visible = den > self.wvn_vis_min
            labeled = self._path_cell_mask(path)
            near = self._near_mask(path)
            gate = torch.zeros_like(visible)
            for b in range(B):
                # tau reference and gated population share the near mask: score
                # grows with viewing distance, so far path cells would inflate tau
                ref = score_bev[b][labeled[b] & visible[b] & near[b]]
                if ref.numel() >= self.wvn_min_cells:
                    tau = torch.quantile(ref, self.wvn_tau_q)
                    gate[b] = visible[b] & ~labeled[b] & near[b] & (score_bev[b] > tau)
        return gate, score_bev, den

    def _path_cell_mask(self, path):
        """Cells covered by any labeled path state (incl. harvested extras and
        zero-weight ones: traversed is traversed), dilated by one cell."""
        B = path.shape[0]
        X, Y = self.map_size
        finite = torch.isfinite(path[..., 0])
        i = torch.round(self.map_origin[0] - path[..., 0] / self.map_resolution[0]).long()
        j = torch.round(self.map_origin[1] - path[..., 1] / self.map_resolution[1]).long()
        ok = finite & (i >= 0) & (i < X) & (j >= 0) & (j < Y)
        mask = torch.zeros(B, X, Y, dtype=torch.float32, device=path.device)
        b = torch.arange(B, device=path.device).unsqueeze(1).expand_as(i)
        mask[b[ok], i[ok], j[ok]] = 1.0
        return F.max_pool2d(mask.unsqueeze(1), 3, 1, 1).squeeze(1) > 0

    def _near_mask(self, path):
        """Cells within MAX_DIST of the robot: the gate was fit on 1-6 m views,
        far-field ground scores OOD from viewing distance alone."""
        B = path.shape[0]
        X, Y = self.map_size
        dev = path.device
        xs = (self.map_origin[0] - torch.arange(X, device=dev, dtype=torch.float32)) * self.map_resolution[0]
        ys = (self.map_origin[1] - torch.arange(Y, device=dev, dtype=torch.float32)) * self.map_resolution[1]
        origin = path[:, 0, :2]
        origin = torch.where(torch.isfinite(origin), origin, torch.zeros_like(origin))
        dx = xs.view(1, X, 1) - origin[:, 0].view(B, 1, 1)
        dy = ys.view(1, 1, Y) - origin[:, 1].view(B, 1, 1)
        return dx.square() + dy.square() <= self.wvn_max_dist**2

    def training_step(self, batch, batch_idx):
        (color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics, path,
         target_trav, trav_weights, depth_target, depth_mask, image_feats) = batch

        depth_raw = depth_voxel
        if self.pcloud_dropout > 0:
            drop = torch.rand(depth_voxel.shape[0], device=depth_voxel.device) < self.pcloud_dropout
            if drop.any():
                depth_voxel = depth_voxel.clone()
                depth_voxel[drop] = 0

        trav_map, mu_logits, pred_depth, debug = self.model(
            color_img, depth_voxel, intrinsics_full, inv_intrinsics, extrinsics,
            depth_target, image_feats=image_feats)

        trav_loss, _ = self.trav_criterion(path, trav_map, target_trav, trav_weights, mu_logits)
        self.mu_metrics.update_train(self, path, trav_map, target_trav)

        # WVN pull-to-zero on gated cells (needs the unflattened depth target)
        gate, score_bev, den = self._wvn_gate(
            image_feats, inv_intrinsics, extrinsics, depth_target, depth_mask, path)

        depth_target = depth_target.view(-1, *depth_target.shape[2:])
        depth_mask = depth_mask.view(-1, *depth_mask.shape[2:])
        depth_loss = self.depth_criterion(pred_depth, depth_target, depth_mask)
        n_gate = gate.sum()
        wvn_loss = (trav_map[:, 0].square() * gate).sum() / n_gate.clamp(min=1)

        loss = trav_loss + self.wvn_weight * wvn_loss
        if self.train_depth:
            loss = loss + self.depth_weight * depth_loss

        if (batch_idx % self.vis_interval) == 0:
            pcloud = self._voxelize_for_vis(color_img, depth_voxel, intrinsics_full, extrinsics)
            self.visualize(color_img, pcloud, trav_map, pred_depth, depth_target, depth_mask,
                           debug, path, inv_intrinsics, extrinsics, target_trav, trav_weights,
                           mu_logits, depth_voxel=depth_raw, intrinsics_full=intrinsics_full,
                           prefix='train')
            score_vis = self._bev_overlays(score_bev.unsqueeze(1), path, normalize=True)
            gate_vis = self._bev_overlays(gate.unsqueeze(1).float(), path)
            den_vis = self._bev_overlays(den.unsqueeze(1), path, normalize=True)
            self.logger.log_image(
                key="train/wvn_gate",
                images=[im.numpy() for im in self._hstack(score_vis, den_vis, gate_vis)])

        self.log("train/loss", loss, on_step=True, on_epoch=False)
        self.log("train/trav_loss", trav_loss, on_step=True, on_epoch=False)
        self.log("train/depth_loss", depth_loss, on_step=True, on_epoch=False)
        self.log("train/wvn_loss", wvn_loss, on_step=True, on_epoch=False)
        self.log("train/wvn_gate_frac", n_gate / gate.numel(), on_step=True, on_epoch=False)
        self.log("train/wvn_vis_frac", (den > self.wvn_vis_min).float().mean(), on_step=True, on_epoch=False)
        self.log("train/depth_mae", adaptations.depth_mae(self.grid_bounds, pred_depth, depth_target, depth_mask), on_step=True, on_epoch=False)
        self.log("train/mu_spread", adaptations.mu_spread(trav_map), on_step=True, on_epoch=False)

        return loss
