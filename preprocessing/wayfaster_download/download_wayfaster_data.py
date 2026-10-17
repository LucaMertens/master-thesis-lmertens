"""Download the alphasense, dlio and tf_minimal bags per GrandTour mission via kleinkram."""

import subprocess
import sys
from pathlib import Path

import kleinkram
from shared_utils.paths import ROS1_ROOT
from data_labeling_pipeline.grandtour_missions import get_mission_by_shortname, missions

INCLUDE_MISSIONS = {m["short_name"] for m in missions}

EXCLUDE_MISSIONS = {"ICE-1"}

TARGET_MISSIONS = sorted(INCLUDE_MISSIONS - EXCLUDE_MISSIONS)


PATTERNS = ["*alphasense*", "*dlio*", "*tf_minimal*"]


def get_files(mission_shortname: str):
    info = get_mission_by_shortname(mission_shortname)
    matches = kleinkram.list_missions(
        project_names=["GrandTourDataset"],
        mission_names=[f"release_{info['timestamp']}"],
    )
    if not matches:
        print(f"  WARNING: mission {mission_shortname} not found in kleinkram")
        return []
    return kleinkram.list_files(mission_ids=[matches[0].id], file_names=PATTERNS)


def prompt_download():
    total_bytes = 0
    for mission in TARGET_MISSIONS:
        files = get_files(mission)
        print(f"\n{mission}: {len(files)} files")
        for f in files:
            print(f"  {f.name}  ({f.size / 1e9:.2f} GB)")
        total_bytes += sum(f.size for f in files)
    print(f"\nTotal: {total_bytes / 1e9:.2f} GB")
    return input("Proceed? [y/N] ").strip().lower() in {"y", "yes"}


def download():
    for mission in TARGET_MISSIONS:
        files = get_files(mission)
        if not files:
            continue
        dest = ROS1_ROOT / mission
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading {mission} → {dest}")
        kleinkram.download(
            file_ids=[f.id for f in files],
            dest=str(dest),
            verbose=False,
            overwrite=False,
        )


if __name__ == "__main__":
    if "--background" in sys.argv:
        download()
    else:
        if prompt_download():
            log_file = Path("download_wayfaster.log")
            subprocess.Popen(
                f"TQDM_DISABLE=1 nohup uv run python -u {__file__} --background > {log_file} 2>&1 &",
                shell=True,
            )
            print(f"Download started in background. Tailing {log_file}...")
            try:
                subprocess.run(["tail", "-f", str(log_file)], check=False)
            except KeyboardInterrupt:
                print("\nExiting tail. Download continues in background.")
