"""Download ap20_prism_position and zed2i_vio bags for missions with ap20 GT (odometry_eval.ipynb).

Lists missing files by default; pass --download to fetch.
"""

import sys

import kleinkram
from shared_utils.paths import ROS1_ROOT
from data_labeling_pipeline.grandtour_missions import missions
from rosbags.rosbag1 import Reader

PATTERNS = {"ap20_prism": "*ap20_prism_position*", "zed2i_vio": "*zed2i_vio*"}


def remote_files(short_name, mission_timestamp):
    m = kleinkram.list_missions(
        project_names=["GrandTourDataset"],
        mission_names=[f"release_{mission_timestamp}"],
    )
    if not m:
        return {}
    out = {}
    for k, pat in PATTERNS.items():
        files = kleinkram.list_files(mission_ids=[m[0].id], file_names=[pat])
        if files and files[0].size > 0:
            out[k] = files[0]
    return out


def local_file(short_name, key):
    return next((ROS1_ROOT / short_name).glob(f"*{key}*.bag"), None)


def is_valid(path, expected_size):
    """Size matches and bag index is readable (catches partial or corrupt files)."""
    if path.stat().st_size != expected_size:
        return False
    try:
        with Reader(path) as r:
            _ = r.connections
        return True
    except Exception:
        return False


def plan():
    total, missing = 0, 0
    for m in missions:
        sn = m["short_name"]
        remote = remote_files(sn, m["timestamp"])
        if "ap20_prism" not in remote:
            continue  # no GT
        for k, f in remote.items():
            local = local_file(sn, k)
            if local is None:
                state, missing, total = "MISSING", missing + 1, total + f.size
            elif not is_valid(local, f.size):
                state, missing, total = (
                    "CORRUPT -> refetch",
                    missing + 1,
                    total + f.size,
                )
            else:
                state = "ok"
            if state != "ok":
                print(f"{sn:8s} {k:10s} {f.size / 1e6:6.2f} MB  {state}")
    print(f"\n{missing} files to (re)fetch, {total / 1e6:.2f} MB")


def download():
    for m in missions:
        sn = m["short_name"]
        remote = remote_files(sn, m["timestamp"])
        if "ap20_prism" not in remote:
            continue
        for k, f in remote.items():
            local = local_file(sn, k)
            if local is not None and is_valid(local, f.size):
                continue
            if local is not None:  # corrupt or partial
                print(f"deleting corrupt {local.name}", flush=True)
                local.unlink()
            dest = ROS1_ROOT / sn
            dest.mkdir(parents=True, exist_ok=True)
            print(f"\nDownloading {sn}/{k} -> {dest}", flush=True)
            kleinkram.download(
                file_ids=[f.id], dest=str(dest), verbose=False, overwrite=False
            )


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    else:
        plan()
