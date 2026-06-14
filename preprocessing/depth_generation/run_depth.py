"""Shell out to depth_generation/run.sh (Docker) for a mission. Needs a Docker + GPU host."""

import argparse
import os
import pathlib
import shutil
import subprocess

from config.paths import EXTRACTED_ROOT

DEPTH_REPO = pathlib.Path(__file__).parent


def run_depth(missions: list[str], output_root, gpus: str = "all",
              depth_repo=DEPTH_REPO, force: bool = False) -> dict[str, str]:
    """Process all `missions` in a single container, one GPU per mission at a time.

    Returns a dict of {mission: error} for missions that failed inside the container.
    """
    depth_repo = pathlib.Path(depth_repo)
    run_sh = depth_repo / "run.sh"
    if shutil.which("docker") is None:
        raise RuntimeError("docker not found. Run depth generation on a Docker+GPU host.")
    if not run_sh.exists():
        raise FileNotFoundError(f"{run_sh} not found.")

    env = dict(os.environ)
    env["GPUS"] = gpus
    env["DATA_DIR"] = str(pathlib.Path(output_root).resolve())
    cmd = ["bash", str(run_sh), "--missions", *(f"/data/{m}" for m in missions)]
    if force:
        cmd.append("--force")
    print(f"+ {' '.join(cmd)}   (GPUS={gpus}, DATA_DIR={env['DATA_DIR']})")

    failures: dict[str, str] = {}
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
        if line.startswith("!! ") and " failed: " in line:
            mission_path, err = line[len("!! "):].rstrip("\n").split(" failed: ", 1)
            failures[pathlib.Path(mission_path).name] = err
    returncode = proc.wait()
    if returncode != 0 and not failures:
        raise subprocess.CalledProcessError(returncode, cmd)
    return failures


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mission", nargs="+", required=True)
    p.add_argument("--output-root", type=pathlib.Path, default=EXTRACTED_ROOT)
    p.add_argument("--gpus", default="all")
    p.add_argument("--force", action="store_true", help="Re-run even if up to date")
    args = p.parse_args()
    run_depth(args.mission, args.output_root, args.gpus, force=args.force)


if __name__ == "__main__":
    main()
