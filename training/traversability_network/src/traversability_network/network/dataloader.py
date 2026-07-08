# Taken from wayfaster repository (https://github.com/matval/wayfaster)
# Used as the base Dataset; GrandTourDataset subclasses it (see grandtour_dataset.py).
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

# Horizontal flip mirrors the base frame (y -> -y) and the camera x-axis (the
# image u-flip), so cam2base conjugates as FLIP_BASE @ cam2base @ FLIP_CAM.
FLIP_BASE = np.diag([1.0, -1.0, 1.0, 1.0])
FLIP_CAM = np.diag([-1.0, 1.0, 1.0, 1.0])


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
        self.harvest_extra = getattr(configs.DATASET, "HARVEST_LOWMU_EXTRA", 0)
        self.harvest_time_window = getattr(configs.DATASET, "HARVEST_TIME_WINDOW", 0.0)

        self.bin_width = 0.1
        self.dtype = torch.float32

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
                # Append to the list of rosbag data
                if len(formatted_data["image_timestamp"]) >= self.n_frames:
                    self.rosbags.append(formatted_data)
                    self.bag_sizes.append(
                        self.bag_sizes[-1]
                        + len(formatted_data["image_timestamp"])
                        - self.n_frames
                        + 1
                    )

        if weights is None:
            # Get dataset statistics
            self.weights, self.bins = self.prepare_weights()
        else:
            self.weights = weights
            self.bins = np.linspace(0, 1, int(1 / self.bin_width) + 1)

        # Flat list of physical window indices (identity here; subclasses may
        # replicate entries to oversample, see GrandTourDataset).
        self.indices = list(range(self.bag_sizes[-1]))

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
            # Get current data
            image_timestamp = rosbag_dict["image_timestamp"][data_idx + t].copy()
            color_fname = rosbag_dict["color_image"][data_idx + t]
            depth_fname = rosbag_dict["depth_image"][data_idx + t]
            # pcloud_fname = rosbag_dict['point_cloud'][data_idx].copy()
            intrinsics = rosbag_dict["intrinsics"][data_idx + t].copy()
            cam2base = rosbag_dict["cam2base"][data_idx + t].copy()

            # Read RGB data
            color_img = cv2.imread(color_fname, -1)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            # Read depth data
            depth_img = cv2.imread(depth_fname, -1)
            # lidar_img = cv2.imread(lidar_fname, -1)

            if horizontal_flip:
                color_img = np.fliplr(color_img).copy()
                depth_img = np.fliplr(depth_img).copy()
                # K is native-res; depth stays native even when color is pre-resized.
                # fliplr sends pixel u to (W - 1) - u.
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
        # Apply augmentation as the outermost transform on the BEV-frame extrinsics,
        # matching how get_states augments label positions (aug_R @ p_anchor + aug_t).
        # Baking aug into cam2base instead (pre-get_extrinsics) would sandwich it
        # between the body pitch/roll and yaw/position transforms, which only agrees
        # with the label-side augmentation when those are identity/commuting -- false
        # in general for a legged robot on uneven terrain.
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
            # Scale K from native to input resolution. Depth is always native-res
            # (needed for the GPU voxelization), color may already be pre-resized.
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

                # Create empty frustum voxel (do not reuse the depth_voxel name:
                # it would clobber the cleaned depth stack returned above)
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

        # mu/nu observability gates: precomputed smoothstep from the RHE (0 at standstill
        # where the parameter is unobservable, 1 in normal motion). A soft mask, never a
        # general scalar reweighting.
        trav_weights[:, 0] *= mu_observability
        trav_weights[:, 1] *= nu_observability

        # Visibility gate: zero the weight of horizon cells no input frame can see.
        # They are unpredictable from the input, and their gradient only pulls the
        # prediction toward the label mean.
        if self.visibility_gate:
            trav_weights *= self.visible_states(
                states, intrinsics_list, extrinsics_list
            )[:, None]

        # Because we have a single camera
        color_img = torch.stack(color_img_list)
        # One transform call on the whole (T, C, H, W) stack: jitter parameters are
        # sampled once, so all frames of a window get the same photometric change
        # instead of flickering frame to frame.
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

        return (
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

        map_float = lambda x: np.array(list(map(float, x)))

        for i in range(len(images_data)):
            timestamp = images_data["timestamp"].iloc[i]
            color_fname = images_data["image"].iloc[i]
            depth_fname = images_data["depth"].iloc[i]
            intrinsics = images_data["intrinsics"].iloc[i]
            cam2base = images_data["cam2base"].iloc[i]

            # pcloud_fname = os.path.join(curr_dir, pcloud_fname)
            img_root = curr_dir
            if getattr(self.configs.DATASET, "RESIZED_ROOT", None):
                img_root = os.path.join(
                    self.configs.DATASET.RESIZED_ROOT, os.path.basename(curr_dir)
                )
            color_fname = os.path.join(img_root, color_fname)

            depth_root = curr_dir
            if getattr(self.configs.DATASET, "DEPTH_CACHE_ROOT", None):
                depth_root = os.path.join(
                    self.configs.DATASET.DEPTH_CACHE_ROOT, os.path.basename(curr_dir)
                )
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
            # Read orientation quaternion (x, y, z, w; used as-is by get_states/get_extrinsics)
            quaternion = quaternion[1:-1].split()
            quaternion = map_float(quaternion)
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
            "position": position_list,
            "quaternion": quaternion_list,
            "action": action_list,
            "traversability": trav_list,
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
        curr_euler_angle = curr_rotation.as_euler("zyx")

        # Get rotations from quaternions
        rotation = R.from_quat(quaternion)
        # Get euler angles from rotations
        euler_angle = rotation.as_euler("zyx")
        # Get extrinsics disregarding heading angle
        heading_rot = R.from_euler("zyx", [curr_euler_angle[0], 0, 0])
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

        # Ground height at each state in the anchor frame: relative base z plus the
        # base-to-ground offset from the stance feet. Flip and the yaw-only aug leave
        # z untouched. z=0 itself is the base origin, ~0.55 m above the ground.
        if "ground_z" in rosbag_dict:
            ground = np.asarray(rosbag_dict["ground_z"])[sync]
        else:
            ground = np.zeros(len(sync))
        ground = position[:, 2] + ground

        states = np.hstack(
            (position[:, :2], euler_angle[:, 0:1], action, ground[:, None])
        )

        # Harvest: extra low-mu states from anywhere in the bag that land inside
        # this window's grid, so each rare low-mu cell is supervised from every
        # window that can see it, not only along the 30s future path.
        if self.train and self.harvest_extra and len(rosbag_dict.get("lowmu_idx", ())):
            extra = self._harvest_lowmu(
                rosbag_dict, timestamps, curr_position, heading_rot,
                aug_transformation, horizontal_flip,
            )
            if extra is not None:
                states = np.concatenate((states, extra[0]))
                traversability = np.concatenate((traversability, extra[1]))
                mu_observability = np.concatenate((mu_observability, extra[2]))
                nu_observability = np.concatenate((nu_observability, extra[3]))

        if horizontal_flip:
            states[:, 4] = -states[:, 4]

        return states, traversability, mu_observability, nu_observability

    def _harvest_lowmu(
        self, rosbag_dict, timestamps, curr_position, heading_rot, aug, horizontal_flip
    ):
        """Low-mu states of the bag outside this window's horizon, transformed to the
        anchor frame and filtered to the grid. The visibility gate downstream still
        decides whether an input frame actually sees them."""
        st = np.asarray(rosbag_dict["states_timestamp"])
        idx = rosbag_dict["lowmu_idx"]
        on_path = (st[idx] >= timestamps[0]) & (st[idx] <= timestamps[-1])
        idx = idx[~on_path]
        if self.harvest_time_window > 0:
            in_window = np.abs(st[idx] - timestamps[0]) <= self.harvest_time_window
            idx = idx[in_window]
        if len(idx) == 0:
            return None
        position = np.asarray(rosbag_dict["position"])[idx]
        position = (heading_rot.inv().as_matrix() @ (position - curr_position).T).T
        if horizontal_flip:
            position[:, 1] = -position[:, 1]
        position = (aug[:3, :3] @ position.T).T + aug[:3, 3]
        xb, yb = self.grid_bounds["xbound"], self.grid_bounds["ybound"]
        inside = (
            (position[:, 0] >= xb[0]) & (position[:, 0] < xb[1])
            & (position[:, 1] >= yb[0]) & (position[:, 1] < yb[1])
        )
        idx, position = idx[inside], position[inside]
        if len(idx) == 0:
            return None
        if len(idx) > self.harvest_extra:
            pick = np.random.choice(len(idx), self.harvest_extra, replace=False)
            idx, position = idx[pick], position[pick]
        yaw = R.from_quat(np.asarray(rosbag_dict["quaternion"])[idx]).as_euler("zyx")[:, :1]
        if horizontal_flip:
            yaw = -yaw
        yaw = yaw + np.arctan2(aug[1, 0], aug[0, 0])
        action = np.asarray(rosbag_dict["action"])[idx]
        if "ground_z" in rosbag_dict:
            ground = np.asarray(rosbag_dict["ground_z"])[idx]
        else:
            ground = np.zeros(len(idx))
        ground = position[:, 2] + ground
        states = np.hstack((position[:, :2], yaw, action, ground[:, None]))
        return (
            states,
            np.asarray(rosbag_dict["traversability"])[idx],
            np.asarray(rosbag_dict["mu_observability"])[idx],
            np.asarray(rosbag_dict["nu_observability"])[idx],
        )

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
        # Get euler angles from rotations
        euler_angle = rotation.as_euler("zyx")
        # Get extrinsics disregarding heading angle
        heading_rot = R.from_euler("zyx", [euler_angle[-1, 0], 0, 0])
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
                # Full mirror conjugation; only negating the y translation leaves
                # the mount misalignment un-mirrored (~15 px projection error).
                cam2base = FLIP_BASE @ cam2base @ FLIP_CAM

            # Get extrinsics disregarding heading angle
            base_rot = R.from_euler("zyx", [0, euler_angle[i, 1], euler_angle[i, 2]])
            base_trans = np.eye(4)
            base_trans[:3, :3] = base_rot.as_matrix()
            odom_trans = np.eye(4)
            odom_rot = R.from_euler(
                "zyx", [euler_angle[i, 0] - euler_angle[-1, 0], 0, 0]
            )
            odom_trans[:3, :3] = odom_rot.as_matrix()
            odom_trans[:2, 3] = position[i, :2]
            extrinsics = odom_trans @ base_trans @ cam2base
            extrinsics_list.append(extrinsics)

        return extrinsics_list

    def visible_states(self, states, intrinsics_list, extrinsics_list):
        """1.0 per horizon state whose ground point falls inside at least one input
        frame's frustum (within the dbound depth range), else 0.0. The intrinsics
        must already be scaled to the input size."""
        d0, d1 = self.grid_bounds["dbound"][0], self.grid_bounds["dbound"][1]
        pts = np.column_stack((states[:, :2], states[:, 5]))
        visible = np.zeros(len(states), dtype=bool)
        for K, ext in zip(intrinsics_list, extrinsics_list):
            cam = (pts - ext[:3, 3]) @ ext[:3, :3]
            z = cam[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = K[0, 0] * cam[:, 0] / z + K[0, 2]
                v = K[1, 1] * cam[:, 1] / z + K[1, 2]
            visible |= (
                (z > d0) & (z < d1)
                & (u >= 0) & (u < self.image_size[0])
                & (v >= 0) & (v < self.image_size[1])
            )
        return visible.astype(np.float64)

    def read_pcloud(self, depth_image_list, cam_intrinsics_list, cam_extrinsics_list):
        """
        Read and process point cloud data from depth images.

        Args:
            depth_image_list (list): List of depth images.
            cam_intrinsics_list (list): List of camera intrinsics matrices.
            cam_extrinsics_list (list): List of camera extrinsics matrices.

        Returns:
            numpy.ndarray: Processed point cloud data.
        """
        temporal_grid = []
        for t in range(len(depth_image_list)):
            depth_image = depth_image_list[t]
            cam_intrinsics = cam_intrinsics_list[t]
            cam_extrinsics = cam_extrinsics_list[t]
            # separate rotation and translation from extrinsics matrix
            rotation, translation = cam_extrinsics[:3, :3], cam_extrinsics[:3, 3]
            # Create points from depth in the image space
            xs = np.arange(
                0, depth_image.shape[1]
            )  # , fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
            ys = np.arange(
                0, depth_image.shape[0]
            )  # , fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
            xs, ys = np.meshgrid(xs, ys)
            # Camera to ego reference frame
            points = np.stack(
                (
                    xs.flatten() * depth_image.flatten() * 1e-3,
                    ys.flatten() * depth_image.flatten() * 1e-3,
                    depth_image.flatten() * 1e-3,
                ),
                -1,
            )
            points = points[np.isfinite(points[:, 2])]
            points = points[points[:, 2] > 0]
            combined_transformation = rotation @ np.linalg.inv(cam_intrinsics)
            points = (combined_transformation @ points.T).T
            points += translation

            dx = np.asarray([
                row[2]
                for row in [
                    self.grid_bounds["xbound"],
                    self.grid_bounds["ybound"],
                    self.grid_bounds["zbound"],
                ]
            ])
            cx = np.asarray([
                np.round(row[1] / row[2] - 0.5)
                for row in [
                    self.grid_bounds["xbound"],
                    self.grid_bounds["ybound"],
                    self.grid_bounds["zbound"],
                ]
            ]).astype(int)
            nx = np.asarray([
                (row[1] - row[0]) / row[2]
                for row in [
                    self.grid_bounds["xbound"],
                    self.grid_bounds["ybound"],
                    self.grid_bounds["zbound"],
                ]
            ]).astype(int)

            # Create gridmap X x Y x Z
            grid = np.zeros((nx[0], nx[1], nx[2]))
            idx_lidar = np.round(np.array([cx]) - points / dx - 0.5).astype(int)
            idx_lidar = idx_lidar[
                (idx_lidar[:, 0] >= 0)
                * (idx_lidar[:, 0] < nx[0])
                * (idx_lidar[:, 1] >= 0)
                * (idx_lidar[:, 1] < nx[1])
                * (idx_lidar[:, 2] >= 0)
                * (idx_lidar[:, 2] < nx[2])
            ]

            grid[idx_lidar[:, 0], idx_lidar[:, 1], idx_lidar[:, 2]] = 1

            # Transform it to Z x X x Y
            grid = grid.transpose((2, 0, 1))
            temporal_grid.append(grid)

        return np.stack(temporal_grid)

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

        # Smooth out the distribution using a Gaussian kernel. ~0.075 mu span: finer
        # than the old 0.2 (keeps low-mu resolution) but wide enough that a lone rare
        # bin does not blow up its 1/density weight.
        values = gaussian_filter1d(
            values, sigma=0.2 / self.bin_width, axis=0, mode="constant"
        )

        zero_count = (values <= 0).sum()
        if zero_count > 0:
            print(
                f"Warning: {zero_count} bins have zero density, which will lead to infinite weights."
            )

        # Inverse-density LDS, tempered by exponent alpha to soften the rare bins.
        weights = (1.0 / values) ** self.lds_alpha

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
