#!/usr/bin/env python3

# Copied from https://github.com/leggedrobotics/grand_tour_dataset/blob/main/examples_ros1/grand_tour_ros1/scripts/extract_transforms_from_tf_static.py

"""
extract_transforms_from_tf_static.py
Extracts static transforms between two frames from a ROS 1 bag file containing /tf_static messages.
This script searches for a static transform between a specified reference (parent) frame and target (child) frame
within a ROS 1 bag file. It supports outputting the transform in several formats, including YAML, JSON, a 4x4 matrix,
or a NumPy .npz file. The script can also handle inverted transforms (i.e., if the transform is stored in the opposite direction).
Dependencies:
    - numpy
    - pyquaternion
    - pyyaml
    - rosbag (ROS 1 Python API)
    ./extract_static_transform.py bagfile base_frame child_frame [--format yaml|json|matrix|npz] [--output OUTPUT_FILE] [--npz-file NPZ_FILE]
Arguments:
    bagfile           Path to the ROS 1 bag file (.bag) containing /tf_static messages.
    base_frame        The reference (parent) frame name (e.g., "base_link").
    child_frame       The target (child) frame name (e.g., "velodyne").
Options:
    --format          Output format: 'yaml' (default), 'json', 'matrix', or 'npz'.
    --output          Output file for YAML or JSON formats. If not specified, prints to stdout.
    --npz-file        Output filename for npz format (default: "transform.npz").
Supported Output Formats:
    - yaml:   Outputs the transform as a YAML dictionary (default).
    - json:   Outputs the transform as a JSON dictionary.
    - matrix: Outputs the transform as a 4x4 extrinsic matrix (suitable for OpenCV, etc.), along with translation and quaternion.
    - npz:    Saves the transform as a NumPy .npz file containing the matrix, translation, and quaternion.
Behavior:
    - Searches for a direct transform (parent -> child). If not found, searches for the inverse (child -> parent) and inverts it.
    - Warns if multiple matching transforms are found and uses the first one.
    - Checks quaternion normalization and warns if not normalized.
    - Lists all available frames if no matching transform is found.
Examples:
    1. Extract as YAML (default, prints to stdout):
        ./extract_static_transform.py mydata.bag base_link velodyne
    2. Extract as JSON and save to file:
        ./extract_static_transform.py mydata.bag base_link velodyne --format json --output transform.json
    3. Extract as a 4x4 matrix (prints to stdout):
        ./extract_static_transform.py mydata.bag base_link velodyne --format matrix
    4. Extract as a NumPy .npz file:
        ./extract_static_transform.py mydata.bag base_link velodyne --format npz --npz-file my_transform.npz
    5. If the transform is stored in the opposite direction (velodyne -> base_link), the script will automatically invert it.
    6. If no transform is found, the script will print an error and list all available frames in the bag.
Notes:
    - Frame names are compared ignoring leading slashes.
    - The script requires ROS 1 Python environment (for rosbag).
    - Quaternion normalization is checked, and a warning is issued if the quaternion is not unit length.


python3 extract_transforms.py 2024-11-14-12-01-26_tf_model.bag livox_imu livox_lidar --format matrix

which is equivalent to:

rosrun tf tf_echo livox_imu livox_lidar

"""

import argparse
import json
import os
import sys

import numpy as np
import rosbag
import yaml
from pyquaternion import Quaternion

# pip install numpy pyquaternion pyyaml


def check_quaternion(q, rtol=1e-6, atol=1e-8):
    """Return True if ‖q‖ ≈ 1 (Hamilton convention, scalar term w)."""
    norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
    return np.isclose(norm_sq, 1.0, rtol=rtol, atol=atol)


def normalize_quaternion(q, eps=1e-12):
    """Return a *new* pyquaternion.Quaternion that is unit length."""
    v = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Quaternion has (near‑)zero length")
    v /= n
    return Quaternion(v[0], v[1], v[2], v[3])


def frames_match(tr, parent, child):
    """Check if transform connects requested frames, ignoring leading slashes."""
    p_frame = tr.header.frame_id.lstrip("/")
    c_frame = tr.child_frame_id.lstrip("/")
    return p_frame == parent.lstrip("/") and c_frame == child.lstrip("/")


def frames_match_inverted(tr, parent, child):
    """Check if transform is the inverse of what we want."""
    p_frame = tr.header.frame_id.lstrip("/")
    c_frame = tr.child_frame_id.lstrip("/")
    return p_frame == child.lstrip("/") and c_frame == parent.lstrip("/")


def load_static_transform_all(bag_path, parent, child):
    """Search all /tf_static transforms for direct and inverted matches.

    Returns:
        direct: List of transforms matching parent->child
        inverse: List of transforms matching child->parent
        all_frames: List of all frame IDs found in the bag
    """
    direct, inverse = [], []
    all_frames = set()

    if not os.path.exists(bag_path):
        raise FileNotFoundError(f"Bag file does not exist: {bag_path}")
    try:
        with rosbag.Bag(bag_path, "r") as bag:
            for _, msg, _ in bag.read_messages(topics=["/tf_static"]):
                for tr in msg.transforms:
                    # Add frames to our set of all frames
                    all_frames.add(tr.header.frame_id.lstrip("/"))
                    all_frames.add(tr.child_frame_id.lstrip("/"))

                    if frames_match(tr, parent, child):
                        direct.append(tr)
                    elif frames_match_inverted(tr, parent, child):
                        inverse.append(tr)
    except Exception as e:
        raise RuntimeError(f"Failed to read bag: {e}")

    return direct, inverse, sorted(list(all_frames))


def stamp_to_dict(stamp):
    # If stamp has .secs and .nsecs
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return {"secs": int(stamp.secs), "nsecs": int(stamp.nsecs)}
    # If stamp is int/float (seconds)
    elif isinstance(stamp, (float, int)):
        secs = int(stamp)
        nsecs = int((float(stamp) - secs) * 1e9)
        return {"secs": secs, "nsecs": nsecs}
    # Fallback: all zero
    else:
        return {"secs": 0, "nsecs": 0}


def transform_to_dict(ts):
    return {
        "header": {
            "frame_id": ts.header.frame_id,
            "stamp": stamp_to_dict(ts.header.stamp),
        },
        "child_frame_id": ts.child_frame_id,
        "translation": {
            "x": ts.transform.translation.x,
            "y": ts.transform.translation.y,
            "z": ts.transform.translation.z,
        },
        "rotation": {
            "x": ts.transform.rotation.x,
            "y": ts.transform.rotation.y,
            "z": ts.transform.rotation.z,
            "w": ts.transform.rotation.w,
        },
    }


def transform_to_matrix(tr, invert=False):
    """Convert ROS transform to 4x4 matrix. Invert if needed."""
    t = np.array(
        [
            tr.transform.translation.x,
            tr.transform.translation.y,
            tr.transform.translation.z,
        ],
        dtype=np.float64,
    )
    q = Quaternion(
        tr.transform.rotation.w,
        tr.transform.rotation.x,
        tr.transform.rotation.y,
        tr.transform.rotation.z,
    )
    q = q.normalised
    R = q.rotation_matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    if invert:
        T = np.linalg.inv(T)
        # Also recalculate t and q from T for output
        R = T[:3, :3]
        t = T[:3, 3]
        q = Quaternion(matrix=R)
    return T, t, q.elements


def output_matrix(T, t, q, file=None):
    print("# 4x4 transform matrix:")
    np.set_printoptions(precision=6, suppress=True)
    print(T)
    print("# Translation:", t)
    print("# Quaternion [x y z w]:", q)


def output_npz(T, t, q, file="transform.npz"):
    np.savez(file, matrix=T, translation=t, quaternion=q)
    print(f"Saved: {file}")


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Extract a static transform from a ROS1 bag."
    )
    parser.add_argument("bag", help="path to .bag file")
    parser.add_argument("reference_frame", help="parent frame (e.g. base_link)")
    parser.add_argument("target_frame", help="child frame (e.g. velodyne)")
    parser.add_argument(
        "--output", help="Output file (e.g. transform.yaml, transform.json)"
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "json", "matrix", "npz"],
        default="yaml",
        help="Output format",
    )
    parser.add_argument(
        "--npz-file", default="transform.npz", help="Filename for npz output"
    )
    args = parser.parse_args(argv)

    try:
        direct, inverse, all_frames = load_static_transform_all(
            args.bag, args.reference_frame, args.target_frame
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    matches = []
    match_type = None
    invert = False

    if direct:
        matches = direct
        match_type = "direct"
        invert = False
    elif inverse:
        matches = inverse
        match_type = "inverse"
        invert = True

    if not matches:
        print(
            f"Error: No static transform found between '{args.reference_frame}' and '{args.target_frame}' "
            "in either direction.",
            file=sys.stderr,
        )
        print(f"Available frames: {', '.join(all_frames)}", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(
            f"Warning: Multiple static transforms found for {match_type} direction, using the first.",
            file=sys.stderr,
        )

    tr = matches[0]

    # Quaternion sanity and normalization
    if not check_quaternion(tr.transform.rotation):
        print("Warning: Quaternion not normalized.", file=sys.stderr)
    # q_arr = normalize_quaternion(tr.transform.rotation)

    # Output format selection
    if args.format == "yaml":
        out = transform_to_dict(tr)
        if args.output:
            with open(args.output, "w") as f:
                yaml.safe_dump(out, stream=f, default_flow_style=False, sort_keys=False)
            print(f"Saved YAML to {args.output}", file=sys.stderr)
        else:
            yaml.safe_dump(
                out, stream=sys.stdout, default_flow_style=False, sort_keys=False
            )
    elif args.format == "json":
        out = transform_to_dict(tr)
        json_str = json.dumps(out, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(json_str + "\n")
            print(f"Saved JSON to {args.output}", file=sys.stderr)
        else:
            print(json_str)
    elif args.format == "matrix":
        T, t, q = transform_to_matrix(tr, invert=invert)
        output_matrix(T, t, q)
    elif args.format == "npz":
        T, t, q = transform_to_matrix(tr, invert=invert)
        output_npz(T, t, q, file=args.npz_file)
    else:
        print(f"Error: Unknown format {args.format}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
