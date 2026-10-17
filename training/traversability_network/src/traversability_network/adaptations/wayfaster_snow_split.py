"""Build a snow / no-snow OOD split of the WayFASTER dataset (symlink trees with filtered rosbags.csv).

Snow appears in exactly three RealSense recordings (Feb 2022).

    uv run python -m traversability_network.adaptations.wayfaster_snow_split
"""

import sys
from pathlib import Path

SRC = Path("/datahome/lmertens/wayfaster_dataset")
DST = Path("/datahome/lmertens/wayfaster_ood")

SNOW = {
    "realsense/data_train": ["ts_merged_2022_02_04_15h24m42s", "ts_merged_2022_02_10_14h14m34s"],
    "realsense/data_valid": ["ts_merged_2022_02_11_13h20m05s"],
}


def bags(split):
    return [ln.strip() for ln in (SRC / split / "rosbags.csv").read_text().splitlines()[1:] if ln.strip()]


def link(out, entries):
    """entries: (source split, bag name) pairs."""
    out.mkdir(parents=True, exist_ok=True)
    for split, bag in entries:
        dst = out / bag
        if not dst.is_symlink():
            dst.symlink_to(SRC / split / bag)
    (out / "rosbags.csv").write_text("rosbags\n" + "".join(f"{b}\n" for _, b in entries))
    print(f"{out}: {len(entries)} recordings")


def main():
    for split, snow in SNOW.items():
        keep = [(split, b) for b in bags(split) if b not in snow]
        link(DST / "nosnow" / split, keep)
    link(DST / "snow" / "all", [(s, b) for s, bs in SNOW.items() for b in bs])
    link(DST / "snow" / "valid_only", [("realsense/data_valid", b) for b in SNOW["realsense/data_valid"]])


if __name__ == "__main__":
    sys.exit(main())
