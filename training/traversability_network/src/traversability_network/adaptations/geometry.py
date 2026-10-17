"""Flip conjugation and RHE-derived ground height, absent from the original pipeline."""
import numpy as np

# horizontal flip mirrors base y and camera x
FLIP_BASE = np.diag([1.0, -1.0, 1.0, 1.0])
FLIP_CAM = np.diag([-1.0, 1.0, 1.0, 1.0])


def flip_cam2base(cam2base):
    """Full mirror conjugation (negating y translation alone misses the mount rotation)."""
    return FLIP_BASE @ cam2base @ FLIP_CAM


def wayfaster_quat_to_xyzw(quaternion):
    """WayFASTER stores w,x,y,z; scipy wants x,y,z,w."""
    return quaternion[[1, 2, 3, 0]]


def subsample_indices(indices, fraction, seed=0):
    """Seeded random subsample for data-amount ablations (wayfaster loader only)."""
    rng = np.random.default_rng(seed)
    keep = rng.permutation(len(indices))[: max(1, int(round(len(indices) * fraction)))]
    return [indices[i] for i in sorted(keep)]


def keep_recording_indices(bag_sizes, fraction, seed=0):
    """Duration-matched subsample that keeps whole recordings instead of thinning all of them."""
    n_bags = len(bag_sizes) - 1
    target = bag_sizes[-1] * fraction
    kept, count = [], 0
    for bag in np.random.default_rng(seed).permutation(n_bags):
        if count >= target:
            break
        kept.append(bag)
        count += bag_sizes[bag + 1] - bag_sizes[bag]
    return [i for bag in sorted(kept) for i in range(bag_sizes[bag], bag_sizes[bag + 1])]


def ground_height(rosbag_dict, idx, position):
    """Ground z in the anchor frame: base z plus the RHE base-to-ground offset (0 if absent)."""
    if "ground_z" in rosbag_dict:
        ground = np.asarray(rosbag_dict["ground_z"])[idx]
    else:
        ground = np.zeros(len(idx))
    return position[:, 2] + ground
