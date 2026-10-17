"""Rolling resume checkpoint and run-id bookkeeping."""

import os

from pytorch_lightning.callbacks import ModelCheckpoint

RUNS_DIR = "runs"


class ResumeCheckpoint(ModelCheckpoint):
    """One last.ckpt, overwritten at every validation.

    Separate class so the callback state key differs from the archive checkpoint's."""

    def __init__(self, dirpath):
        super().__init__(
            dirpath=dirpath,
            filename="last",
            save_top_k=1,
            every_n_epochs=1,
            enable_version_counter=False,
        )


def _marker(ckpt_base, tag):
    return os.path.join(ckpt_base, RUNS_DIR, tag)


def record_run(ckpt_base, tag, run_id):
    """Remember which wandb run this TAG last launched."""
    path = _marker(ckpt_base, tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        fh.write(run_id)
    os.replace(tmp, path)


def lookup_run(ckpt_base, tag, run_id=None):
    """(wandb id, last.ckpt) to continue: the given id, else the TAG's last run. Either may be None."""
    if run_id is None:
        path = _marker(ckpt_base, tag)
        if not os.path.exists(path):
            return None, None
        with open(path) as fh:
            run_id = fh.read().strip()
    ckpt = os.path.join(ckpt_base, run_id, "last.ckpt")
    return run_id, ckpt if os.path.exists(ckpt) else None
