# Base Dataset from the wayfaster repository (https://github.com/matval/wayfaster).
# GrandTourDataset (adaptations/grandtour_dataset.py) subclasses it.
import bisect
import os
import random
from itertools import compress

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as DataLoader
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from .. import adaptations


class Dataset(DataLoader.Dataset):
    """
    Custom Dataset for handling image and state data.
    """

    def __init__(self, configs, data_path, transform=None, weights=None, train=False):
        """
        Initialize the Dataset object.

        Args:
            configs (object): Configuration object containing dataset parameters.
            data_path (list): List of paths to the dataset directories.
            transform (callable, optional): A function/transform to apply to the data. Defaults to None.
            weights (numpy.ndarray, optional): Precomputed weights for the dataset. Defaults to None.
            train (bool, optional): Whether the dataset is for training. Defaults to False.
        """
        print("Initializing dataset...")
        self.configs = configs
        self.transform = transform
        self.train = train
        self.dt = configs.TRAINING.DT
        self.horizon = configs.TRAINING.HORIZON
        self.image_size = configs.MODEL.INPUT_SIZE
        self.downsample = configs.MODEL.DOWNSAMPLE
        self.verbose = configs.TRAINING.VERBOSE
        self.grid_bounds = configs.MODEL.GRID_BOUNDS
        self.downsample = configs.MODEL.DOWNSAMPLE
        self.horiz_flip = configs.AUGMENTATIONS.HORIZ_FLIP
        self.pcloud_droput = configs.AUGMENTATIONS.PCLOUD_DROPOUT
        self.n_frames = configs.MODEL.TIME_LENGTH
        self.predict_depth = configs.MODEL.PREDICT_DEPTH
        self.max_translation_aug = configs.AUGMENTATIONS.MAX_TRANSLATION
        self.max_rotation_aug = configs.AUGMENTATIONS.MAX_ROTATION
        self.lds_alpha = configs.DATASET.LDS_ALPHA
        self.visibility_gate = getattr(configs.DATASET, "VISIBILITY_GATE", False)
        self.occlusion_gate = getattr(configs.DATASET, "OCCLUSION_GATE", False)
        self.harvest_extra = getattr(configs.DATASET, "HARVEST_LOWMU_EXTRA", 0)
        self.harvest_time_window = getattr(configs.DATASET, "HARVEST_TIME_WINDOW", 0.0)
        # ADAPTATION: temporal sampling of the input window (see adaptations/sampling.py)
        self.frame_stride = getattr(configs.DATASET, "FRAME_STRIDE", 1)
        self.window_stride = getattr(configs.DATASET, "WINDOW_STRIDE", 1)

        self.bin_width = 0.1
        self.dtype = torch.float32
        self.use_dino_cache = False  # GrandTourDataset enables this in DINO mode
        self.emit_mission_id = False  # GrandTourDataset enables this, missions are its concept

        self.map_size = (
            int(
                (self.grid_bounds["xbound"][1] - self.grid_bounds["xbound"][0])
                / self.grid_bounds["xbound"][2]
            ),
            int(
                (self.grid_bounds["ybound"][1] - self.grid_bounds["ybound"][0])
                / self.grid_bounds["ybound"][2]
            ),
        )

        self.rosbags = []
        self.bag_sizes = [0]

        for curr_path in data_path:
            # Get csv file path
            csv_path = os.path.join(curr_path, configs.DATASET.CSV_FILE)
            # Read lines in csv file
            bags_list = pd.read_csv(csv_path)
            for bag in bags_list.values:
                if self.verbose:
                    print("reading bag:", bag[0])

                curr_dir = os.path.join(curr_path, bag[0])
                states_data = pd.read_csv(os.path.join(curr_dir, "states.csv"))
                images_data = pd.read_csv(os.path.join(curr_dir, "images.csv"))

                # Prepare data
                formatted_data = self.read_data(states_data, images_data, curr_dir)
                if getattr(configs.MODEL, "BACKBONE", "resnet34") == "dino":
                    # ADAPTATION: frozen-DINO feature cache (see adaptations/dino_cache.py)
                    adaptations.attach_dino_cache(formatted_data, configs.DATASET.DINO_CACHE_ROOT, bag[0])
                    self.use_dino_cache = True
                # Append to the list of rosbag data
                n_windows = adaptations.window_count(
                    len(formatted_data["image_timestamp"]), self.n_frames, self.frame_stride
                )
                if n_windows > 0:
                    self.rosbags.append(formatted_data)
                    self.bag_sizes.append(self.bag_sizes[-1] + n_windows)

        if weights is None:
            # Get dataset statistics
            self.weights, self.bins = self.prepare_weights()
        else:
            self.weights = weights
            self.bins = np.linspace(0, 1, int(1 / self.bin_width) + 1)

        # Flat list of physical window indices (subclasses may replicate entries
        # to oversample, see GrandTourDataset).
        # ADAPTATION: WINDOW_STRIDE thins near-duplicate overlapping windows.
        self.indices = adaptations.strided_indices(self.bag_sizes, self.window_stride)

        # ADAPTATION: data-amount ablation, deterministic train-only subsample.
        seed = getattr(configs.DATASET, "TRAIN_SUBSET_SEED", 0)
        frac = getattr(configs.DATASET, "TRAIN_FRACTION", 1.0)
        if self.train and frac < 1.0:
            self.indices = adaptations.subsample_indices(self.indices, frac, seed)
        rec_frac = getattr(configs.DATASET, "TRAIN_RECORDING_FRACTION", 1.0)
        if self.train and rec_frac < 1.0:
            self.indices = adaptations.keep_recording_indices(self.bag_sizes, rec_frac, seed)

        if self.verbose:
            print("weights:", self.weights)
            print("bins:", self.bins)

        print("Dataset initialized!")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns
        -------
            data: list of [color_img, lidar_img, inv_intrinsics, extrinsics, states, traversability]:
                color_img: torch.Tensor<float> (3, H, W)
                    normalised color image.
                lidar_img: torch.Tensor<float> (1, H, W)
                    normalised depth image.
                intrinsics: torch.Tensor<float> (3, 3)
                    camera's intrinsics matrix.
                extrinsics: torch.Tensor<float> (4, 4)
                    6 DoF pose from world coordinates to camera coordinates.
                states: torch.Tensor<float> (300, 6)
                    array containing (x, y, theta, v, omega, ground z)
                traversability: torch.Tensor<float> (300, 2)
                    array containing (mu, nu)
        """
        # Map the logical index to a physical window (oversampling replicates entries)
        idx = self.indices[idx]
        # Find the rosbag that contains this data index
        bag_idx = bisect.bisect_right(self.bag_sizes, idx) - 1
        # Get the respective rosbag data
        rosbag_dict = self.rosbags[bag_idx]
        # Get sequence index inside the rosbag
        data_idx = idx - self.bag_sizes[bag_idx]
        # ADAPTATION: per-mission FRAME_STRIDE (adaptations/grandtour_dataset.py)
        frame_stride = rosbag_dict.get("frame_stride", self.frame_stride)

        # Augment with horizontal flip
        if self.train:
            horizontal_flip = random.random() < self.horiz_flip
            aug_transformation = self.sample_transformation()
        else:
            horizontal_flip = False
            aug_transformation = np.eye(4)

        image_timestamp_list = []
        color_img_list = []
        depth_img_list = []
        intrinsics_list = []
        cam2base_list = []
        for t in range(self.n_frames):
            # Get current data. ADAPTATION: FRAME_STRIDE spaces the stacked frames out
            f = data_idx + t * frame_stride
            image_timestamp = rosbag_dict["image_timestamp"][f].copy()
            color_fname = rosbag_dict["color_image"][f]
            depth_fname = rosbag_dict["depth_image"][f]
            # pcloud_fname = rosbag_dict['point_cloud'][data_idx].copy()
            intrinsics = rosbag_dict["intrinsics"][f].copy()
            cam2base = rosbag_dict["cam2base"][f].copy()

            # Read RGB data. ADAPTATION: raw .npy sidecar instead of PNG decode when
            # DATASET.NPY_SIDECARS (see adaptations/npy_cache.py)
            sidecars = getattr(self, "use_npy_sidecars", True)
            color_img = adaptations.imread(color_fname, sidecars)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            # Read depth data
            depth_img = adaptations.imread(depth_fname, sidecars)
            # lidar_img = cv2.imread(lidar_fname, -1)

            if horizontal_flip:
                color_img = np.fliplr(color_img).copy()
                depth_img = np.fliplr(depth_img).copy()
                # K is native-res (depth stays native), fliplr maps u to (W - 1) - u
                intrinsics[0, 2] = depth_img.shape[1] - 1 - intrinsics[0, 2]

            image_timestamp_list.append(image_timestamp)
            color_img_list.append(color_img)
            depth_img_list.append(depth_img)
            intrinsics_list.append(intrinsics)
            cam2base_list.append(cam2base)

        # Read states and traversability information already with horizontal flip
        states, traversability, mu_observability, nu_observability = self.get_states(
            rosbag_dict, image_timestamp_list, aug_transformation, horizontal_flip
        )
        extrinsics_list = self.get_extrinsics(
            rosbag_dict, image_timestamp_list, cam2base_list, horizontal_flip
        )
        # Augment the BEV-frame extrinsics outermost, matching the label-side augmentation.
        # Baking it into cam2base would not commute with body pitch/roll on a legged robot.
        extrinsics_list = [aug_transformation @ e for e in extrinsics_list]
        # Clean full-res depth + intrinsics for GPU voxelization (grid built in the model)
        depth_voxel, intrinsics_full = self.read_pcloud(
            depth_img_list, intrinsics_list, extrinsics_list
        )

        depth_target_list = []
        depth_mask_list = []
        for t in range(self.n_frames):
            color_img = color_img_list[t]
            depth_img = depth_img_list[t]
            # Scale K to input resolution (depth is always native-res)
            intrinsics_list[t][:1] *= self.image_size[0] / depth_img.shape[1]
            intrinsics_list[t][1:2] *= self.image_size[1] / depth_img.shape[0]
            if (color_img.shape[1], color_img.shape[0]) != tuple(self.image_size):
                color_img = cv2.resize(
                    color_img, self.image_size, interpolation=cv2.INTER_AREA
                )
            if (depth_img.shape[1], depth_img.shape[0]) != tuple(self.image_size):
                depth_img = cv2.resize(
                    depth_img, self.image_size, interpolation=cv2.INTER_NEAREST
                )

            color_img_list[t] = (
                torch.from_numpy(color_img).permute(2, 0, 1).type(self.dtype)
                / 255.0
            )
            """
             Here, if the network is predicting the depth, then the target_depth is the training label
             Otherwise, the target_depth is the voxel representation of the deterministic depth
             to be used instead of the depth prediction
            """
            # Convert depth image to target image
            depth_target = depth_img * 1e-3
            depth_target[~np.isfinite(depth_target)] = 0
            depth_size = (
                np.ceil(self.image_size[0] / self.downsample).astype("int"),
                np.ceil(self.image_size[1] / self.downsample).astype("int"),
            )
            depth_target = cv2.resize(
                depth_target, depth_size, interpolation=cv2.INTER_NEAREST
            )
            # Get depth mask
            depth_mask = (depth_target != 0.0).astype(np.float64)
            # Discretize depth
            depth_target = np.round(
                (depth_target - self.grid_bounds["dbound"][0])
                / self.grid_bounds["dbound"][2]
            ).astype("int")
            n_d = int(
                (self.grid_bounds["dbound"][1] - self.grid_bounds["dbound"][0])
                / self.grid_bounds["dbound"][2]
            )

            if self.predict_depth:
                # Handle edge cases
                depth_mask[depth_target < 0] = 0
                depth_target[depth_target < 0] = 0
                depth_mask[depth_target > n_d - 1] = 0
                depth_target[depth_target > n_d - 1] = n_d - 1
            else:
                # Get downsampled depth size
                fH = int(np.ceil(self.image_size[1] / self.downsample))
                fW = int(np.ceil(self.image_size[0] / self.downsample))
                # Create points from depth in the image space
                xs = np.arange(0, fH)
                ys = np.arange(0, fW)
                xs, ys = np.meshgrid(xs, ys, indexing="ij")
                # Camera to ego reference frame
                points = np.stack(
                    (depth_target.flatten(), xs.flatten(), ys.flatten()), -1
                )

                # Not named depth_voxel: that would clobber the depth stack returned above
                frustum = np.zeros((n_d, fH, fW))
                # Filter undesirable values
                idxs = (points[:, 0] >= 0) & (points[:, 0] <= n_d - 1)
                points = points[idxs]
                # And assign ones to populates depths
                frustum[points[:, 0], points[:, 1], points[:, 2]] = 1.0
                depth_target = frustum.astype("float32")

            depth_target_list.append(depth_target)
            depth_mask_list.append(depth_mask)

        # Get traversability weights
        weight_idxs = np.digitize(traversability, self.bins[:-1]) - 1
        trav_weights = np.zeros_like(traversability)
        trav_weights[:, 0] = self.weights[weight_idxs[:, 0], 0]
        trav_weights[:, 1] = self.weights[weight_idxs[:, 1], 1]

        # ADAPTATION: mu/nu observability gate (RHE smoothstep, 0 at standstill)
        trav_weights = adaptations.observability_gate(trav_weights, mu_observability, nu_observability)

        # ADAPTATION: visibility/occlusion gate (zero cells no input frame can see)
        if self.visibility_gate:
            trav_weights = adaptations.apply_visibility_gate(
                trav_weights, states, intrinsics_list, extrinsics_list,
                self.grid_bounds, self.image_size, self.occlusion_gate, depth_img_list,
            )

        # Because we have a single camera
        color_img = torch.stack(color_img_list)
        # One call on the (T, C, H, W) stack: same jitter for all frames of a window
        if self.transform is not None:
            color_img = self.transform(color_img)
        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)
        depth_target = np.stack(depth_target_list)
        depth_mask = np.stack(depth_mask_list)

        # Pad states, traversability and weights in case they are short
        # (total length includes the harvested low-mu slots, empty when unused)
        total = self.horizon + self.harvest_extra
        pad = np.ones([total - states.shape[0], states.shape[1]]) * np.inf
        states = np.concatenate((states, pad))
        pad = np.zeros([
            total - traversability.shape[0],
            traversability.shape[1],
        ])
        traversability = np.concatenate((traversability, pad))
        pad = np.zeros([total - trav_weights.shape[0], trav_weights.shape[1]])
        trav_weights = np.concatenate((trav_weights, pad))

        # Attribute data type
        color_img = color_img.type(self.dtype)
        depth_voxel = torch.from_numpy(depth_voxel)  # int16 mm, voxelized on the GPU
        intrinsics_full = torch.from_numpy(intrinsics_full).type(self.dtype)
        intrinsics = torch.from_numpy(intrinsics).type(self.dtype)
        extrinsics = torch.from_numpy(extrinsics).type(self.dtype)
        states = torch.from_numpy(states).type(self.dtype)
        traversability = torch.from_numpy(traversability).type(self.dtype)
        trav_weights = torch.from_numpy(trav_weights).type(self.dtype)
        depth_target = torch.from_numpy(depth_target)
        depth_mask = torch.from_numpy(depth_mask).type(self.dtype)

        out = (
            color_img,
            depth_voxel,
            intrinsics_full,
            intrinsics,
            extrinsics,
            states,
            traversability,
            trav_weights,
            depth_target,
            depth_mask,
        )
        if self.use_dino_cache:
            # ADAPTATION: frozen-DINO feature cache (see adaptations/dino_cache.py)
            feats = adaptations.read_dino_feats(
                rosbag_dict, data_idx, self.n_frames, horizontal_flip, frame_stride
            )
            out = out + (feats,)
        # ADAPTATION: mission index for per-mission validation metrics
        # (see adaptations/mission_ids.py, which owns this order)
        if self.emit_mission_id:
            out = out + (adaptations.emit_mission_id(rosbag_dict),)
        # ADAPTATION: human W-RIZZ pair supervision, always last
        # (see adaptations/human_pairs.py)
        if getattr(self, "human", None) is not None:
            out = out + (self.human.emit(
                bag_idx, data_idx, aug_transformation, horizontal_flip),)
        return out

    def read_data(self, states_data, images_data, curr_dir):
        """
        Read and prepare data from CSV files.

        Args:
            states_data (pd.DataFrame): DataFrame containing states data.
            images_data (pd.DataFrame): DataFrame containing images data.
            curr_dir (str): Current directory path.

        Returns:
            dict: Dictionary containing prepared data.
        """
        # Images file
        image_timestamp_list = []
        color_fname_list = []
        depth_fname_list = []
        intrinsics_list = []
        cam2base_list = []
        # States file
        states_timestamp_list = []
        position_list = []
        quaternion_list = []
        action_list = []
        trav_list = []

        map_float = lambda x: np.array(list(map(float, x)))  # noqa: E731

        # ADAPTATION: resized-image / depth-cache root fallback (see adaptations/paths.py)
        img_root, depth_root = adaptations.resolve_media_roots(
            curr_dir,
            getattr(self.configs.DATASET, "RESIZED_ROOT", None),
            getattr(self.configs.DATASET, "DEPTH_CACHE_ROOT", None),
        )

        for i in range(len(images_data)):
            timestamp = images_data["timestamp"].iloc[i]
            color_fname = images_data["image"].iloc[i]
            depth_fname = images_data["depth"].iloc[i]
            intrinsics = images_data["intrinsics"].iloc[i]
            cam2base = images_data["cam2base"].iloc[i]

            # pcloud_fname = os.path.join(curr_dir, pcloud_fname)
            color_fname = os.path.join(img_root, color_fname)
            depth_fname = os.path.join(depth_root, depth_fname)

            # Read intrinsics matrix
            intrinsics = intrinsics.replace("[", " ").replace("]", " ").split()
            intrinsics = map_float(intrinsics).reshape((3, 3))

            # Read extrinsics matrix
            cam2base = cam2base.replace("[", " ").replace("]", " ").split()
            cam2base = map_float(cam2base).reshape((4, 4))

            # Append values to lists
            image_timestamp_list.append(timestamp)
            color_fname_list.append(color_fname)
            depth_fname_list.append(depth_fname)
            intrinsics_list.append(intrinsics)
            cam2base_list.append(cam2base)

        for timestamp, position, quaternion, action, traversability in states_data.iloc:
            # Read linear control action
            position = position[1:-1].split()
            position = map_float(position)
            # Read orientation quaternion. ADAPTATION: the CSV stores it w,x,y,z;
            # reorder to x,y,z,w (scipy convention) so get_states/get_extrinsics use
            # it as-is. GrandTour's own loader already stores x,y,z,w.
            quaternion = quaternion[1:-1].split()
            quaternion = adaptations.wayfaster_quat_to_xyzw(map_float(quaternion))
            # Read control action
            action = action[1:-1].split()
            action = map_float(action)
            # Read traversability
            traversability = traversability[1:-1].split()
            traversability = map_float(traversability)

            # Append values to lists
            states_timestamp_list.append(timestamp)
            position_list.append(position)
            quaternion_list.append(quaternion)
            action_list.append(action)
            trav_list.append(traversability)

        # Remove data to make all the images have a future horizon
        idxs = np.asarray(image_timestamp_list) < states_timestamp_list[-1]
        image_timestamp_list = list(compress(image_timestamp_list, idxs))
        color_fname_list = list(compress(color_fname_list, idxs))
        depth_fname_list = list(compress(depth_fname_list, idxs))
        intrinsics_list = list(compress(intrinsics_list, idxs))
        cam2base_list = list(compress(cam2base_list, idxs))

        if self.verbose:
            print(
                "All data have been loaded from bag! Total dataset size: {:d}".format(
                    len(color_fname_list)
                )
            )

        data_dict = {
            "image_timestamp": image_timestamp_list,
            "color_image": color_fname_list,
            "depth_image": depth_fname_list,
            "intrinsics": intrinsics_list,
            "cam2base": cam2base_list,
            "states_timestamp": states_timestamp_list,
            # ndarrays, not lists: np.asarray per sample was ~70% of dataloader time
            "position": np.asarray(position_list),
            "quaternion": np.asarray(quaternion_list),
            "action": np.asarray(action_list),
            "traversability": np.asarray(trav_list),
            # The CSV labels have no RHE observability gates: fully observable.
            "mu_observability": np.ones(len(states_timestamp_list)),
            "nu_observability": np.ones(len(states_timestamp_list)),
        }

        return data_dict

    def get_states(
        self, rosbag_dict, image_timestamp, aug_transformation, horizontal_flip=False
    ):
        """
        Get states synchronized to the time horizon.

        Args:
            rosbag_dict (dict): Dictionary containing rosbag data.
            image_timestamp (list): List of image timestamps.
            horizontal_flip (bool, optional): Whether to apply horizontal flip. Defaults to False.

        Returns:
            tuple: States and traversability.
        """
        timestamps = np.arange(0, self.horizon) * self.dt + image_timestamp[0]
        # Get states synchronized to the time horizon
        sync = [
            bisect.bisect_left(rosbag_dict["states_timestamp"], i)
            for i in timestamps
            if i < rosbag_dict["states_timestamp"][-1]
        ]
        position = np.asarray(rosbag_dict["position"])[sync]
        quaternion = np.asarray(rosbag_dict["quaternion"])[sync]
        action = np.asarray(rosbag_dict["action"])[sync]
        traversability = np.asarray(rosbag_dict["traversability"])[sync]
        # mu/nu observability gates (1 -> fully observable when absent, e.g. CSV path)
        mu_observability = np.asarray(rosbag_dict["mu_observability"])[sync]
        nu_observability = np.asarray(rosbag_dict["nu_observability"])[sync]

        # Get current position and rotation
        curr_idx = bisect.bisect_left(
            rosbag_dict["states_timestamp"], image_timestamp[-1]
        )
        curr_position = np.asarray(rosbag_dict["position"])[curr_idx]
        curr_quaternion = np.asarray(rosbag_dict["quaternion"])[curr_idx]
        curr_rotation = R.from_quat(curr_quaternion)
        # Intrinsic "ZYX" (capital) = yaw-pitch-roll; lowercase "zyx" is extrinsic
        # (R = Rx Ry Rz) and scrambles the tilt axes on slopes.
        curr_euler_angle = curr_rotation.as_euler("ZYX")

        # Get rotations from quaternions
        rotation = R.from_quat(quaternion)
        # Get euler angles from rotations
        euler_angle = rotation.as_euler("ZYX")
        # Get extrinsics disregarding heading angle
        heading_rot = R.from_euler("ZYX", [curr_euler_angle[0], 0, 0])
        # Transform position in relation to the first timestamp
        position = (
            heading_rot.inv().as_matrix() @ (position.T - curr_position[None].T)
        ).T

        # Mirror before augmenting: __getitem__ composes the extrinsics as
        # aug @ flip(E), so the labels must be aug @ flip(p) to stay consistent.
        if horizontal_flip:
            position[:, 1] = -position[:, 1]
            euler_angle[:, 0] = -euler_angle[:, 0]

        # Apply augmentation
        position = (aug_transformation[:3, :3] @ position.T).T + aug_transformation[
            :3, 3
        ]
        euler_angle[:, 0] += np.arctan2(
            aug_transformation[1, 0], aug_transformation[0, 0]
        )

        # ADAPTATION: ground height from RHE base-to-ground offset (see adaptations/geometry.py)
        # z=0 is the base origin, ~0.55 m above the ground; flip and the yaw-only aug leave z untouched.
        ground = adaptations.ground_height(rosbag_dict, sync, position)

        states = np.hstack(
            (position[:, :2], euler_angle[:, 0:1], action, ground[:, None])
        )

        # ADAPTATION: harvest extra low-mu states from anywhere in the bag (see adaptations/harvest.py)
        if self.train and self.harvest_extra and len(rosbag_dict.get("lowmu_idx", ())):
            extra = adaptations.harvest_lowmu(
                rosbag_dict, timestamps, curr_position, heading_rot, aug_transformation,
                horizontal_flip, self.harvest_extra, self.harvest_time_window, self.grid_bounds,
            )
            if extra is not None:
                states = np.concatenate((states, extra[0]))
                traversability = np.concatenate((traversability, extra[1]))
                mu_observability = np.concatenate((mu_observability, extra[2]))
                nu_observability = np.concatenate((nu_observability, extra[3]))

        if horizontal_flip:
            states[:, 4] = -states[:, 4]

        return states, traversability, mu_observability, nu_observability

    def get_extrinsics(
        self, rosbag_dict, image_timestamp, cam2base_list, horizontal_flip=False
    ):
        """
        Get extrinsics synchronized to the time horizon.

        Args:
            rosbag_dict (dict): Dictionary containing rosbag data.
            image_timestamp (list): List of image timestamps.
            cam2base_list (list): List of camera to base transformations.
            horizontal_flip (bool, optional): Whether to apply horizontal flip. Defaults to False.

        Returns:
            list: List of extrinsics matrices.
        """
        # Get states synchronized to the time horizon
        sync = [
            bisect.bisect_left(rosbag_dict["states_timestamp"], i)
            for i in image_timestamp
            if i < rosbag_dict["states_timestamp"][-1]
        ]
        position = np.asarray(rosbag_dict["position"])[sync]
        quaternion = np.asarray(rosbag_dict["quaternion"])[sync]

        # Get rotations from quaternions
        rotation = R.from_quat(quaternion)
        # Intrinsic yaw-pitch-roll so R = Rz(yaw) @ Ry(pitch) @ Rx(roll); the
        # odom_trans @ base_trans split below relies on exactly this factorization.
        euler_angle = rotation.as_euler("ZYX")
        # Get extrinsics disregarding heading angle
        heading_rot = R.from_euler("ZYX", [euler_angle[-1, 0], 0, 0])
        # Transform position in relation to the first timestamp
        position = (
            heading_rot.inv().as_matrix() @ (position.T - position[-1, None].T)
        ).T

        if horizontal_flip:
            position[:, 1] = -position[:, 1]
            euler_angle[:, 0] = -euler_angle[:, 0]
            euler_angle[:, 2] = -euler_angle[:, 2]

        extrinsics_list = []
        for i, cam2base in enumerate(cam2base_list):
            if horizontal_flip:
                # ADAPTATION: full mirror conjugation fix (see adaptations/geometry.py)
                cam2base = adaptations.flip_cam2base(cam2base)

            # Get extrinsics disregarding heading angle
            base_rot = R.from_euler("ZYX", [0, euler_angle[i, 1], euler_angle[i, 2]])
            base_trans = np.eye(4)
            base_trans[:3, :3] = base_rot.as_matrix()
            odom_trans = np.eye(4)
            odom_rot = R.from_euler(
                "ZYX", [euler_angle[i, 0] - euler_angle[-1, 0], 0, 0]
            )
            odom_trans[:3, :3] = odom_rot.as_matrix()
            odom_trans[:2, 3] = position[i, :2]
            extrinsics = odom_trans @ base_trans @ cam2base
            extrinsics_list.append(extrinsics)

        return extrinsics_list

    def read_pcloud(self, depth_image_list, cam_intrinsics_list, cam_extrinsics_list):
        """
        Clean native-res depth for the GPU voxelization (TravNet.voxelize); same
        contract as GrandTourDataset.read_pcloud, without its rectification
        border crop. Clips depth to the dbound range and returns (depth (T, H, W)
        int16 mm, native intrinsics (T, 3, 3)); extrinsics are consumed GPU-side.
        """
        dmin, dmax = (
            self.grid_bounds["dbound"][0] * 1000,
            self.grid_bounds["dbound"][1] * 1000,
        )
        cleaned, full_K = [], []
        for depth, K in zip(depth_image_list, cam_intrinsics_list):
            depth = depth.copy()
            depth[~np.isfinite(depth)] = 0
            depth[(depth < dmin) | (depth > dmax)] = 0
            cleaned.append(depth.astype(np.int16))
            full_K.append(K.copy())
        return np.stack(cleaned), np.stack(full_K)

    def prepare_weights(self):
        """
        Prepare weights for the dataset based on the traversability distribution.

        Returns:
            tuple: Weights and bins for the dataset.
        """
        # Missions are split into segments at larger time gaps.
        # Each segment of a mission uses the same traversability object (duplicated by reference).
        # It is indexed at access time by timestamps.
        # We only want to count each object once so missions split into many segments don't dominate the density estimate.
        arrays = {
            id(data["traversability"]): data["traversability"] for data in self.rosbags
        }
        label = np.concatenate([np.asarray(a) for a in arrays.values()])

        # Calculate histogram
        values = np.zeros((int(1 / self.bin_width), 2))
        values[:, 0], _ = np.histogram(
            label[:, 0], bins=int(1 / self.bin_width), range=(0, 1), density=True
        )
        values[:, 1], bins = np.histogram(
            label[:, 1], bins=int(1 / self.bin_width), range=(0, 1), density=True
        )

        # Smooth the distribution so a lone rare bin does not blow up its 1/density weight
        values = gaussian_filter1d(
            values, sigma=0.2 / self.bin_width, axis=0, mode="constant"
        )

        zero_count = (values <= 0).sum()
        if zero_count > 0:
            print(
                f"Warning: {zero_count} bins have zero density, which will lead to infinite weights."
            )

        # ADAPTATION: inverse-density LDS tempered by exponent alpha
        weights = adaptations.lds_weights(values, self.lds_alpha)

        return weights, bins

    def sample_transformation(self):
        """
        Sample a random transformation for data augmentation.

        Returns:
            numpy.ndarray: Random transformation matrix.
        """
        # Random rotation
        theta = random.uniform(-self.max_rotation_aug, self.max_rotation_aug)
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

        # Random translation
        translation = np.random.uniform(
            -self.max_translation_aug, self.max_translation_aug, 2
        )

        transformation = np.eye(4)
        transformation[:2, :2] = rotation
        transformation[:2, 3] = translation

        return transformation
