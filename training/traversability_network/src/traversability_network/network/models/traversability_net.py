# Taken from wayfaster repository (https://github.com/matval/wayfaster)
import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_fusion import TemporalModel
from .encoder_decoder import Decoder
from ...adaptations import build_encoder, head_dropout
from ...adaptations import capacity, wrizz_arch

class TravNet(nn.Module):
    """
    TravNet: A neural network for traversability prediction and navigation.

    Attributes:
        grid_bounds (dict): Dictionary containing the boundaries for the grid.
        input_size (tuple): Size of the input image.
        downsample (int): Downsampling factor.
        image_dim (int): Dimension of the image.
        temporal_length (int): Length of the temporal sequence.
        predict_depth (bool): Whether to predict depth or not.
        fuse_pcloud (bool): Whether to fuse point cloud data or not.
    """
    def __init__(
            self, grid_bounds, input_size, downsample=8,
            image_dim=64, temporal_length=3, predict_depth=True,
            fuse_pcloud=True, voxel_downsample=1, mu_bins=0, evidential=False,
            freeze_backbone=False, backbone="resnet34", dino_feat_dim=768,
            dino_neck_dim=256, dino_noise=0.0, mu_pixel_linear=False, head_drop=0.0,
            slim_decoder=False, wrizz_decoder=False
        ):
        super(TravNet, self).__init__()

        self.grid_bounds    = grid_bounds
        self.input_size     = (input_size[1], input_size[0])
        self.camC           = image_dim
        self.downsample     = downsample     # Due to the encoder
        self.predict_depth  = predict_depth
        self.fuse_pcloud    = fuse_pcloud
        self.mu_bins        = mu_bins        # 0 = sigmoid regression head
        self.evidential      = evidential    # mu head emits a Normal-Inverse-Gamma instead
        self.voxel_downsample = voxel_downsample  # nearest-subsample depth before voxelizing
        self.eps            = 1e-6
        
        dx = torch.Tensor([row[2] for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])
        # round, not truncate: (2.6 - -2.0) / 0.2 = 22.999... would silently drop a bin
        nx = torch.LongTensor([round((row[1] - row[0]) / row[2]) for row in [grid_bounds['xbound'], grid_bounds['ybound'], grid_bounds['zbound']]])

        self.int_nx = [int(x) for x in nx.cpu().detach().tolist()]

        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # Images mean and std
        mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float()
        std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

        # Create 3D voxels
        self.voxels = self.create_voxels()

        # Define predicted depth dimension
        if self.predict_depth:
            self.D = int((grid_bounds['dbound'][1] - grid_bounds['dbound'][0]) / grid_bounds['dbound'][2])
        else:
            self.D = 0

        self.latent_dim = image_dim

        # Define pcloud fusion dimension
        if self.fuse_pcloud:
            pcloud_dim = self.nx[2]
        else:
            pcloud_dim = 0
        
        # ADAPTATION: trainable ResNet, or frozen-DINO neck on precomputed image_feats
        self.backbone = backbone
        self.encoder = build_encoder(
            backbone, self.D + self.camC, self.downsample, self.input_size,
            freeze_backbone, dino_feat_dim, dino_neck_dim, dino_noise)

        # Bird's eye view compressor
        # ADAPTATION: 1x1 in the slim variant (see adaptations/capacity.py)
        self.bev_compressor = capacity.bev_compressor(
            self.camC * self.nx[2] + pcloud_dim, self.camC, slim_decoder)

        # Temporal Model
        self.temporal_model = TemporalModel(
            channels=self.camC,
            temporal_length=temporal_length,
            input_shape=(self.int_nx[0], self.int_nx[1]))

        # Bird's eye view decoder
        # ADAPTATION: wrizz_decoder swaps the upsampling blocks (adaptations/wrizz_arch.py)
        self.decoder = Decoder(in_channels=self.latent_dim, slim=slim_decoder)
        if wrizz_decoder:
            wrizz_arch.apply(self.decoder)
        
        # Map head, one mu parameterization: sigmoid regression (2 ch), mu_bins logits + nu,
        # or evidential NIG (5 ch)
        assert not (self.mu_bins and self.evidential), "mu_bins and evidential are mutually exclusive"
        # Linear per-pixel mu probe on frozen DINO features, lifted to BEV with the depth
        # splat; the BEV head then only outputs nu.
        self.mu_pixel_linear = mu_pixel_linear
        if mu_pixel_linear:
            assert backbone == "dino" and self.mu_bins, "mu_pixel_linear needs BACKBONE=dino and MU_BINS>0"
            self.mu_pixel_head = nn.Conv2d(dino_feat_dim, self.mu_bins, kernel_size=1)
        if self.evidential:
            out_ch = 5
        elif self.mu_pixel_linear:
            out_ch = 1
        elif self.mu_bins:
            out_ch = self.mu_bins + 1
        else:
            out_ch = 2
        # ADAPTATION: head_drop injects Dropout2d for regularized ablations
        head_layers = [
            *head_dropout(head_drop),
            nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.latent_dim),
            nn.ReLU(inplace=True),
            *head_dropout(head_drop),
            nn.Conv2d(self.latent_dim, out_ch, kernel_size=1, padding=0)]
        if self.mu_pixel_linear or (not self.mu_bins and not self.evidential):
            head_layers.append(nn.Sigmoid())
        self.travmap_head = nn.Sequential(*head_layers)
        if self.mu_bins:
            centers = (torch.arange(self.mu_bins, dtype=torch.float) + 0.5) / self.mu_bins
            self.mu_centers = nn.Parameter(centers, requires_grad=False)
    
    def create_voxels(self):
        """
        Create a 3D grid in the map space.

        Returns:
            torch.nn.Parameter: 3D grid of voxels.
        """
        grid_z = torch.arange(*self.grid_bounds['zbound'], dtype=torch.float).flip(0)
        grid_z = torch.reshape(grid_z, [self.nx[2], 1, 1])
        grid_z = grid_z.repeat(1, self.nx[0], self.nx[1])

        grid_y = torch.arange(*self.grid_bounds['ybound'], dtype=torch.float).flip(0)
        grid_y = torch.reshape(grid_y, [1, 1, self.nx[1]])
        grid_y = grid_y.repeat(self.nx[2], self.nx[0], 1)

        grid_x = torch.arange(*self.grid_bounds['xbound'], dtype=torch.float).flip(0)
        grid_x = torch.reshape(grid_x, [1, self.nx[0], 1])
        grid_x = grid_x.repeat(self.nx[2], 1, self.nx[1])

        # Z x X x Y x 3
        voxels = torch.stack((grid_x, grid_y, grid_z), -1)
        return nn.Parameter(voxels, requires_grad=False)
    
    def get_inv_geometry(self, intrinsics, extrinsics):
        """
        Calculate the mapping from 3D map voxels to camera frustum.

        Args:
            intrinsics (torch.Tensor): Intrinsics matrix (3x3) for projection.
            extrinsics (torch.Tensor): Extrinsics matrix (4x4) with rotation and translation.

        Returns:
            torch.Tensor: Transformed points in the camera reference frame (u, v, depth).
        """
        # separate rotation and translation from extrinsics matrix
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        # get batch and number of cams dimensions
        B, N, _ = translation.shape
        # Add batch, camera dimension, and a dummy dimension at the end
        points = self.voxels.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        points = points.expand(B, -1, -1, -1, -1, -1, -1)
        # Voxels to camera frame
        points = points - translation.view(B, N, 1, 1, 1, 3, 1)
        # rotation is orthonormal (rigid extrinsics), so its inverse is its transpose
        combined_transformation = intrinsics.matmul(rotation.transpose(-1, -2))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)

        points = torch.cat((points[..., :2] / (points[..., 2:3] + self.eps), points[..., 2:3]), -1)

        # The 3 dimensions in the camera reference frame (u, v, depth)
        return points
    
    def sample2bev(self, geometry, x):
        """
        Sample from the frustum to bird's eye view.

        Args:
            geometry (torch.Tensor): Geometry of the frustum.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Bird's eye view representation.
        """
        # batch*T, depth, height, width, channels
        _, d, h, w, c = x.shape

        batch, T, Z, X, Y, _ = geometry.shape
        geometry = geometry.view(batch*T, Z, X, Y, 3)

        # Normalize grid
        u = 2 * geometry[..., 0] / (self.input_size[1]-1) - 1
        v = 2 * geometry[..., 1] / (self.input_size[0]-1) - 1
        depth = 2 * (geometry[...,2] - self.grid_bounds['dbound'][0]) / (self.grid_bounds['dbound'][1] - self.grid_bounds['dbound'][0]) - 1
        grid = torch.stack((u, v, depth), -1)

        # Sample from frustum
        x = F.grid_sample(x, grid, align_corners=False)
        x = x.view(batch, T, *x.shape[1:])
        return x

    @torch.compiler.disable
    def voxelize(self, depth, intrinsics, extrinsics, downsample=None):
        """Back-project depth into the (Z, X, Y) occupancy grid on the GPU.

        Vectorized port of the CPU read_pcloud in network/dataloader.py. depth is
        metric (m) of shape (BT, H, W), intrinsics the matching full-res K (BT, 3, 3),
        extrinsics the cam2base poses (BT, 4, 4).

        ``downsample`` nearest-subsamples depth by that factor (scaling K to match)
        before back-projection: ~factor^2 fewer points, but thins far-field occupancy
        (precision stays exact, recall drops with range, see check_voxel_downscale.py).
        """
        if downsample is None:
            downsample = self.voxel_downsample
        dev = depth.device
        with torch.autocast(device_type=dev.type, enabled=False):
            depth = depth.float()
            if downsample > 1:
                # strided subsample adds no phantom voxels; K scales by 1/s for the same rays
                depth = depth[:, ::downsample, ::downsample]
                intrinsics = intrinsics.clone()
                intrinsics[:, :2, :] = intrinsics[:, :2, :] / downsample
            BT, H, W = depth.shape
            vs, us = torch.meshgrid(
                torch.arange(H, device=dev, dtype=torch.float),
                torch.arange(W, device=dev, dtype=torch.float), indexing='ij')
            d = depth.reshape(BT, -1)                       # (BT, N)
            u, v = us.reshape(-1), vs.reshape(-1)           # (N,)
            pts = torch.stack((u * d, v * d, d), dim=1)     # (BT, 3, N)
            rotation, translation = extrinsics[:, :3, :3].float(), extrinsics[:, :3, 3:4].float()
            pts = rotation.matmul(torch.inverse(intrinsics.float())).matmul(pts) + translation
            pts = pts.transpose(1, 2)                       # (BT, N, 3)

            cx = torch.tensor(
                [round(b[1] / b[2] - 0.5) for b in
                 (self.grid_bounds['xbound'], self.grid_bounds['ybound'], self.grid_bounds['zbound'])],
                device=dev, dtype=torch.float)
            idx = torch.round(cx - pts / self.dx - 0.5).long()   # (BT, N, 3)
            nx = self.nx
            valid = d > 0
            for a in range(3):
                valid = valid & (idx[..., a] >= 0) & (idx[..., a] < nx[a])

            grid = torch.zeros(BT, int(nx[0] * nx[1] * nx[2]), device=dev)
            flat = idx[..., 0] * (nx[1] * nx[2]) + idx[..., 1] * nx[2] + idx[..., 2]
            b_idx = torch.arange(BT, device=dev).view(BT, 1).expand_as(flat)
            grid[b_idx[valid], flat[valid]] = 1.0
            # (BT, X, Y, Z) -> (BT, Z, X, Y)
            return grid.view(BT, nx[0], nx[1], nx[2]).permute(0, 3, 1, 2)

    def forward(self, color_img, depth_voxel, intrinsics_full, intrinsics, extrinsics, depth_img=None, image_feats=None):
        """
        Forward pass of the TravNet model.

        Args:
            color_img (torch.Tensor): Color image tensor of shape (B, T, C, H, W).
            depth_voxel (torch.Tensor): Full-res metric depth (mm) of shape (B, T, H, W), voxelized on the GPU.
            intrinsics_full (torch.Tensor): Full-res intrinsics matching depth_voxel, shape (B, T, 3, 3).
            intrinsics (torch.Tensor): Input-size intrinsics tensor of shape (B, T, 3, 3).
            extrinsics (torch.Tensor): Extrinsics matrix tensor of shape (B, T, 4, 4).
            depth_img (torch.Tensor, optional): Depth image tensor. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - trav_map (torch.Tensor): Traversability map of shape (B, 2, X, Y), (mu, nu);
                  mu is the bin-expectation for the binned head or gamma for the evidential head.
                - mu_logits (torch.Tensor or None): auxiliary mu tensor, meaning depends on the
                  head: bin logits (B, mu_bins, X, Y) for the binned head, NIG (v, alpha, beta)
                  (B, 3, X, Y) for the evidential head, None for plain regression.
                - depth_logits (torch.Tensor): Predicted depth logits of shape (B*T, D, H, W).
                - debug (torch.Tensor): Debug information tensor.
        """
        B, T, C, imH, imW = color_img.shape
        assert(C==3)

        color_img = color_img.view(B*T, C, imH, imW)

        if image_feats is not None:
            # Precomputed frozen backbone features (already normalized at cache time)
            x = self.encoder(image_feats.view(B*T, *image_feats.shape[2:]).float())
        else:
            # Normalize image according to statistics
            x = (color_img - self.mean) / self.std
            # Pass image throught encoder
            x = self.encoder(x)

        # Depth is B x N x D x H/downsample x W/downsample
        if self.predict_depth:
            depth_logits = x[:, :self.D]
        else:
            depth_logits = depth_img.view(B*T, *depth_img.shape[2:])
        depth_context = x[:, self.D:(self.D + self.camC)]

        # depth_logits is B*N x D x H/downsample x W/downsample
        # depth_context is B*N x C x H/downsample x W/downsample
        x = depth_logits.softmax(dim=1).unsqueeze(1) * depth_context.unsqueeze(2)

        # Create 3D frustrum
        geom = self.get_inv_geometry(intrinsics, extrinsics)

        mu_pixel_logits = None
        if self.mu_pixel_linear:
            assert image_feats is not None
            # Depth weights are detached so the mu loss cannot shape the depth path
            pl = self.mu_pixel_head(F.normalize(image_feats.view(B*T, *image_feats.shape[2:]).float(), p=2, dim=1))
            pl = F.interpolate(pl, size=depth_logits.shape[-2:], mode="bilinear", align_corners=False)
            pl = torch.cat([pl, torch.ones_like(pl[:, :1])], dim=1)
            mu_x = depth_logits.softmax(dim=1).detach().unsqueeze(1) * pl.unsqueeze(2)
            mu_bev = self.sample2bev(geom, mu_x)                      # B x T x K+1 x Z x X x Y
            num = mu_bev[:, :, :self.mu_bins].sum(dim=(1, 3))
            den = mu_bev[:, :, self.mu_bins:].sum(dim=(1, 3))
            mu_pixel_logits = num / (den + self.eps)                  # B x K x X x Y

        x = self.sample2bev(geom, x)

        # X is B x T x C x Z x X x Y ->  B*T x C*Z x X x Y
        x = x.view(B*T, -1, *x.shape[4:])
        debug = x

        if self.fuse_pcloud:
            # Build the occupancy grid on the GPU and concatenate with the frustum features
            pcloud = self.voxelize(
                depth_voxel.view(B*T, *depth_voxel.shape[2:]) * 1e-3,
                intrinsics_full.view(B*T, 3, 3),
                extrinsics.view(B*T, 4, 4))
            x = torch.cat([x, pcloud.to(x.dtype)], dim=1)
        
        # And compress X to B x T x C x X x Y
        x = self.bev_compressor(x)

        # Recover temporal component: B x T x C x X x Y
        x = x.view(B, T, *x.shape[1:])

        # Temporal fusion with 3D conv layers
        x = self.temporal_model(x)
        
        # Now, x is a grid of shape (B x C x X x Y)
        x = x.view(B, -1, self.int_nx[0], self.int_nx[1])
        
        # BEV decoder
        bev_features = self.decoder(x)
        
        # Calculate traversability map
        out = self.travmap_head(bev_features)
        if self.mu_pixel_linear:
            mu = (mu_pixel_logits.softmax(1) * self.mu_centers.view(1, -1, 1, 1)).sum(1, keepdim=True)
            trav_map = torch.cat([mu, out], dim=1)
            mu_logits = mu_pixel_logits
        elif self.mu_bins:
            mu_logits = out[:, :self.mu_bins]
            mu = (mu_logits.softmax(1) * self.mu_centers.view(1, -1, 1, 1)).sum(1, keepdim=True)
            trav_map = torch.cat([mu, out[:, self.mu_bins:].sigmoid()], dim=1)
        elif self.evidential:
            gamma = out[:, 0:1].sigmoid()
            v = F.softplus(out[:, 1:2]) + self.eps
            alpha = F.softplus(out[:, 2:3]) + 1.0 + self.eps
            beta = F.softplus(out[:, 3:4]) + self.eps
            trav_map = torch.cat([gamma, out[:, 4:5].sigmoid()], dim=1)
            mu_logits = torch.cat([v, alpha, beta], dim=1)
        else:
            trav_map, mu_logits = out, None

        return trav_map, mu_logits, depth_logits, debug
