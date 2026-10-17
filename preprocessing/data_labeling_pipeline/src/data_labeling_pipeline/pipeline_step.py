"""Skip-if-done markers and logging for per-mission steps (`<name>_metadata.json`, bump version to redo)."""

import json
import time
from contextlib import contextmanager
from pathlib import Path


def is_step_done(output_dir, name: str, version: int) -> bool:
    meta_path = Path(output_dir) / f"{name}_metadata.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    return meta.get("version") == version


def write_step_done(output_dir, name: str, version: int, **extra) -> None:
    meta_path = Path(output_dir) / f"{name}_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"version": version, **extra}))


@contextmanager
def log_step(name: str, output_dir):
    print(f"[{time.strftime('%H:%M:%S')}] Starting {name}: {output_dir}")
    start = time.time()
    yield
    elapsed = time.time() - start
    print(
        f"[{time.strftime('%H:%M:%S')}] Finished {name}: {output_dir} ({elapsed:.1f}s)"
    )
