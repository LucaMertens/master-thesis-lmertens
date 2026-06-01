import subprocess
import sys
from pathlib import Path
from typing import Set

import kleinkram
from data_labeling_pipeline.grandtour_missions import get_mission_by_shortname, missions

grand_tour_folder = Path("/datahome/lmertens/grand_tour_dataset")
downloaded_missions_shortnames = set([
    "ICE-1",
    "SNOW-2",
])

missions_to_download_shortnames: Set[str] = set([m["short_name"] for m in missions])  # ty:ignore[invalid-assignment]
missions_to_download_shortnames = (
    missions_to_download_shortnames - downloaded_missions_shortnames
)


patterns = ["*anymal_command*", "*cpt7_ie_tc*", "*tf_minimal*"]


def get_files(mission: str):
    info = get_mission_by_shortname(mission)
    matches = kleinkram.list_missions(
        project_names=["GrandTourDataset"],
        mission_names=[f"release_{info['timestamp']}"],
    )
    if not matches:
        return []
    return kleinkram.list_files(mission_ids=[matches[0].id], file_names=patterns)


def prompt_download():
    total = 0
    for mission in missions_to_download_shortnames:
        files = get_files(mission)
        print(f"{mission}: {len(files)} files")
        for f in files:
            print(f"  {f.name} ({f.size / 1e6:.1f} MB)")
        total += len(files)
    return input(f"\nTotal: {total} files. Proceed? [y/N] ").strip().lower() in {
        "y",
        "yes",
    }


def download():
    for mission in missions_to_download_shortnames:
        files = get_files(mission)
        if not files:
            continue
        dest = grand_tour_folder / "ros1" / mission
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {mission}...")
        kleinkram.download(
            file_ids=[f.id for f in files],
            dest=str(dest),
            verbose=True,
            overwrite=False,
        )


if __name__ == "__main__":
    if "--background" in sys.argv:
        download()
    else:
        if prompt_download():
            log_file = Path("download_output.log")
            subprocess.Popen(
                f"nohup uv run python -u {__file__} --background > {log_file} 2>&1 &",
                shell=True,
            )
            print(f"Download started in background. Tailing {log_file}...")
            try:
                subprocess.run(["tail", "-f", str(log_file)], check=False)
            except KeyboardInterrupt:
                print("\nExiting tail. Download continues in background.")
