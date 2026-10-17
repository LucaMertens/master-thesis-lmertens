"""Numbered checkpoint archive on a validation schedule, dense early and sparse later."""

import os

from pytorch_lightning.callbacks import Callback


class ArchiveCheckpoints(Callback):
    """Every validation for the first ``dense_epochs``, every ``every_n``-th after that.

    Weights only by default (50 MB a file instead of 146 MB), since the archive is for
    offline scoring. ModelCheckpoint cannot express the change of rate."""

    def __init__(self, dirpath, dense_epochs, every_n, weights_only=True):
        self.dirpath = dirpath
        self.dense_epochs = dense_epochs
        self.every_n = every_n
        self.weights_only = weights_only
        self.seen = 0

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.seen += 1
        if trainer.current_epoch >= self.dense_epochs and self.seen % self.every_n:
            return
        name = f"val_epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"
        trainer.save_checkpoint(os.path.join(self.dirpath, name), weights_only=self.weights_only)
