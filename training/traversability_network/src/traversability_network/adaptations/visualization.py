"""Observation-only wandb visualization panels, mixed into TrainingModule (uses its self.model/self.grid_bounds/self.logger)."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .overlay import colorize_mu, overlay_bev_on_image, overlay_path_on_image

# Path overlay colors (RGB): red line, green start (robot, t=0), white end (horizon).
PATH_LINE = (255, 0, 0)
PATH_START = (0, 255, 0)
PATH_END = (255, 255, 255)


class VisualizationMixin:
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
        """Log a batch of training/validation panels to wandb under key prefix."""
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
