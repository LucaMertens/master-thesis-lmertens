"""Extract everything derivable from the *_alphasense.bag in one pass.

Writes intrinsics.npy, intrinsics_<cam>.npy, images/<cam>/<ts_ns>.png,
image_manifest.npz (if images), and cam2base.npy, stereo_extrinsics.npz
(if extrinsics).
"""

import argparse
import os
import pathlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

import cv2
import numpy as np
from data_labeling_pipeline.pipeline_step import is_step_done, log_step, write_step_done
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm

from . import rectify

CAMERAS = list(rectify.CAMERAS)

# Bump to invalidate outputs from older versions of these steps.
EXTRINSICS_VERSION = 1
IMAGES_VERSION = 1


def _decode_rectify_write(
    raw: bytes,
    msgtype: str,
    cam: str,
    cam_maps,
    out_dir: pathlib.Path,
    typestore,
    png_compression: int,
) -> tuple[str, int, str]:
    # Runs in a worker thread. cv2 and rosbags both release the GIL.
    msg: Any = typestore.deserialize_ros1(raw, msgtype)
    ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
    img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
    img = rectify.remap(img, cam_maps)
    rel = f"images/{cam}/{ts_ns}.png"
    cv2.imwrite(str(out_dir / rel), img, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    return cam, ts_ns, rel


def _rotate_frame(t_base_cam: np.ndarray, r_raw_rect: np.ndarray) -> np.ndarray:
    """base <- rectified-cam, given base <- raw-cam and (raw -> rect) rotation."""
    out = t_base_cam.copy()
    out[:3, :3] = t_base_cam[:3, :3] @ r_raw_rect.T
    return out


def extract(
    mission_dir: pathlib.Path,
    output_dir: pathlib.Path,
    max_frames: int | None,
    balance: float,
    fov_scale: float,
    images: bool,
    extrinsics: bool,
    num_workers: int | None = None,
    png_compression: int = 3,
    force: bool = False,
) -> None:
    do_extrinsics = extrinsics and (force or not is_step_done(output_dir, "extrinsics", EXTRINSICS_VERSION))
    do_images = images and (force or not is_step_done(output_dir, "images", IMAGES_VERSION))
    if extrinsics and not do_extrinsics:
        print(f"Skipping extrinsics (up to date): {output_dir}")
    if images and not do_images:
        print(f"Skipping images (up to date): {output_dir}")
    if not (do_extrinsics or do_images):
        return

    rect = rectify.build_rectifiers(mission_dir, balance, fov_scale)
    maps = {rectify.REF_CAM: rect.ref, rectify.PARTNER_CAM: rect.partner}
    output_dir.mkdir(parents=True, exist_ok=True)

    if do_extrinsics:
        with log_step("extrinsics", output_dir):
            cam2base = _rotate_frame(
                rect.t_base_ref, rect.R_ref
            )  # base <- rectified front_center
            np.save(output_dir / "cam2base.npy", cam2base)
            np.savez(output_dir / "stereo_extrinsics.npz", baseline=np.array(rect.baseline))
            geom_baseline = float(
                np.linalg.norm(rect.t_base_ref[:3, 3] - rect.t_base_partner[:3, 3])
            )
            print(
                f"baseline: {rect.baseline:.4f} m (from P2) vs {geom_baseline:.4f} m (tf centers)"
            )
            write_step_done(output_dir, "extrinsics", EXTRINSICS_VERSION)

    if not do_images:
        return

    with log_step("images", output_dir):
        bag = next(mission_dir.glob("*_alphasense.bag"), None)
        if bag is None:
            raise FileNotFoundError(f"No *_alphasense.bag in {mission_dir}")

        # Each cv2 call would otherwise spawn its own thread pool, oversubscribing
        # the machine on top of our worker threads.
        cv2.setNumThreads(1)

        np.save(output_dir / "intrinsics.npy", rect.ref.K)
        for cam in CAMERAS:
            np.save(output_dir / f"intrinsics_{cam}.npy", maps[cam].K)
            (output_dir / "images" / cam).mkdir(parents=True, exist_ok=True)

        image_topics = {f"/boxi/alphasense/{c}/image_raw/compressed": c for c in CAMERAS}
        typestore = get_typestore(Stores.ROS1_NOETIC)
        manifest: dict[str, list[tuple[int, str]]] = {c: [] for c in CAMERAS}
        counts = {c: 0 for c in CAMERAS}

        n_workers = num_workers or min(os.cpu_count() or 4, 32)
        # Bound in-flight futures to limit RAM usage.
        max_in_flight = n_workers * 4
        pending: set = set()
        # Keep submission-ordered references for deterministic manifest construction.
        all_futures: list[tuple[str, object]] = []

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            with Reader(bag) as reader:
                conns = [c for c in reader.connections if c.topic in image_topics]
                total = (
                    sum(c.msgcount for c in conns)
                    if max_frames is None
                    else max_frames * len(CAMERAS)
                )
                with tqdm(total=total, desc=mission_dir.name, disable=None) as bar:
                    for conn, _, raw in reader.messages(conns):
                        cam = image_topics[conn.topic]
                        if max_frames is not None and counts[cam] >= max_frames:
                            if all(counts[c] >= max_frames for c in CAMERAS):
                                break
                            continue

                        # Throttle: drain one batch before queuing more.
                        if len(pending) >= max_in_flight:
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            bar.update(len(done))

                        fut = pool.submit(
                            _decode_rectify_write,
                            bytes(raw),
                            conn.msgtype,
                            cam,
                            maps[cam],
                            output_dir,
                            typestore,
                            png_compression,
                        )
                        pending.add(fut)
                        all_futures.append((cam, fut))
                        counts[cam] += 1

                    # Drain whatever is still running.
                    if pending:
                        wait(pending)
                        bar.update(len(pending))

        for cam, fut in all_futures:
            _, ts_ns, rel = fut.result()
            manifest[cam].append((ts_ns, rel))

        np.savez(
            output_dir / "image_manifest.npz",
            **{
                c: np.array(rows, dtype=[("ts_ns", "i8"), ("path", "U128")])
                for c, rows in manifest.items()
            },
        )
        for cam in CAMERAS:
            print(
                f"{cam}: {counts[cam]} frames, K=[{maps[cam].K[0, 0]:.1f}, {maps[cam].K[1, 1]:.1f}]"
            )
        write_step_done(output_dir, "images", IMAGES_VERSION)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mission-dir", required=True, type=pathlib.Path)
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per camera (testing).",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Worker threads (default: min(cpu_count, 32)).",
    )
    p.add_argument(
        "--png-compression",
        type=int,
        default=3,
        choices=range(10),
        metavar="0-9",
        help="PNG compression level (0=fast/large, 9=slow/small).",
    )
    p.add_argument("--balance", type=float, default=rectify.DEFAULT_BALANCE)
    p.add_argument("--fov-scale", type=float, default=rectify.DEFAULT_FOV_SCALE)
    p.add_argument("--no-images", action="store_false", dest="images")
    p.add_argument("--no-extrinsics", action="store_false", dest="extrinsics")
    p.add_argument("--force", action="store_true", help="Re-run even if up to date")
    args = p.parse_args()
    extract(
        args.mission_dir,
        args.output_dir,
        args.max_frames,
        args.balance,
        args.fov_scale,
        args.images,
        args.extrinsics,
        args.jobs,
        args.png_compression,
        args.force,
    )


if __name__ == "__main__":
    main()
