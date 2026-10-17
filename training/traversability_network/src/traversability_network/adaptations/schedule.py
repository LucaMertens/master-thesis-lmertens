"""Cosine LR schedule that survives a resume with a different epoch count."""

import math

from torch.optim.lr_scheduler import CosineAnnealingLR


class CosineSchedule(CosineAnnealingLR):
    """Cosine annealing that keeps the configured T_max on resume and uses the closed form."""

    def load_state_dict(self, state_dict):
        super().load_state_dict({**state_dict, "T_max": self.T_max})

    def get_lr(self):
        return [
            self.eta_min
            + (base - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base in self.base_lrs
        ]
