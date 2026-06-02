"""Shell out to depth_generation/run.sh (Docker) for a mission.

Needs a Docker + GPU host; this devcontainer has none.
"""

import argparse
import os
import pathlib
import shutil
import subprocess

DEPTH_REPO = pathlib.Path(__file__).parents[3] / "depth_generation"


def run_depth(mission: str, output_root, gpus: str = "all",
              models_dir=None, depth_repo=DEPTH_REPO) -> None:
    depth_repo = pathlib.Path(depth_repo)
    run_sh = depth_repo / "run.sh"
    if shutil.which("docker") is None:
        raise RuntimeError(
            "docker not found. Run depth generation on a Docker+GPU host "
            "(this devcontainer has none)."
        )
    if not run_sh.exists():
        raise FileNotFoundError(f"{run_sh} not found.")

    env = dict(os.environ)
    env["GPUS"] = gpus
    env["DATA_DIR"] = str(pathlib.Path(output_root).resolve())
    env["MODELS_DIR"] = str(models_dir or depth_repo / "pretrained_models")
    cmd = ["bash", str(run_sh), "--missions", f"/data/{mission}"]
    print(f"+ {' '.join(cmd)}   (GPUS={gpus}, DATA_DIR={env['DATA_DIR']})")
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mission", required=True)
    p.add_argument("--output-root", type=pathlib.Path, default=pathlib.Path("output"))
    p.add_argument("--gpus", default="all")
    p.add_argument("--models-dir", type=pathlib.Path, default=None)
    args = p.parse_args()
    run_depth(args.mission, args.output_root, args.gpus, args.models_dir)


if __name__ == "__main__":
    main()
