"""Download *dlio* bags for every mission that has a valid cpt7_ie_tc on disk.

Used by odometry_fallback.ipynb to compare dlio against cpt7-tc across all missions.
Run as a single instance: concurrent runs race on the same files and corrupt them.

  uv run python download_dlio.py            # show plan
  uv run python download_dlio.py --download  # verify existing, then fetch missing
"""

import sys
from pathlib import Path

import kleinkram
from rosbags.rosbag1 import Reader
from data_labeling_pipeline.grandtour_missions import get_mission_by_shortname, missions

GT = Path("/datahome/lmertens/grand_tour_dataset")
MIN_TC_BYTES = 1_000_000


def has_valid_tc(short_name):
    d = GT / "ros1" / short_name
    tc = list(d.glob("*cpt7_ie_tc*.bag"))
    return bool(tc) and tc[0].stat().st_size > MIN_TC_BYTES


def dlio_files(short_name):
    info = get_mission_by_shortname(short_name)
    m = kleinkram.list_missions(
        project_names=["GrandTourDataset"], mission_names=[f"release_{info['timestamp']}"]
    )
    if not m:
        return []
    return kleinkram.list_files(mission_ids=[m[0].id], file_names=["*dlio*"])


TARGET = sorted(m["short_name"] for m in missions if has_valid_tc(m["short_name"]))


def local_dlio(short_name):
    return next((GT / "ros1" / short_name).glob("*dlio*.bag"), None)


def is_valid(path, expected_size):
    """Cheap integrity check: correct size and a readable bag index (catches partial / corrupt)."""
    if expected_size and path.stat().st_size != expected_size:
        return False
    try:
        with Reader(path) as r:
            _ = r.connections
        return True
    except Exception:
        return False


def plan():
    total, missing = 0, 0
    for sn in TARGET:
        files = dlio_files(sn)
        size = sum(f.size for f in files)
        local = local_dlio(sn)
        if local is None:
            state, missing, total = "MISSING", missing + 1, total + size
        elif files and not is_valid(local, files[0].size):
            state, missing, total = "CORRUPT -> refetch", missing + 1, total + size
        else:
            state = "ok"
        print(f"{sn:8s} {size / 1e9:5.2f} GB  {state}")
    print(f"\n{len(TARGET)} missions, {missing} to (re)fetch, {total / 1e9:.2f} GB")


def download():
    for sn in TARGET:
        files = dlio_files(sn)
        if not files:
            continue
        local = local_dlio(sn)
        if local is not None and is_valid(local, files[0].size):
            continue
        if local is not None:  # corrupt / partial: remove so it is re-fetched
            print(f"deleting corrupt {local.name}", flush=True)
            local.unlink()
        dest = GT / "ros1" / sn
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading {sn} -> {dest}", flush=True)
        kleinkram.download(file_ids=[f.id for f in files], dest=str(dest), verbose=False, overwrite=False)


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    else:
        plan()
