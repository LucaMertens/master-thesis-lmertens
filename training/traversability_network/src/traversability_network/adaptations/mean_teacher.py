"""Mean teacher (Tarvainen & Valpola 2017) for W-RIZZ: weight-EMA copy with a dense consistency term.

Geometric augmentation is shared, photometric augmentation is sampled independently.
"""

import copy

import torch
import torch.nn as nn
import torchvision.transforms as T


class MeanTeacher(nn.Module):
    """EMA copy of the student, plus the consistency term between the two."""

    def __init__(self, model, decay=0.999, ramp_steps=2000, jitter=(0.4, 0.4, 0.4, 0.1)):
        super().__init__()
        self.teacher = copy.deepcopy(model).requires_grad_(False).eval()
        self.decay = decay
        self.ramp_steps = ramp_steps
        self.jitter = T.ColorJitter(*jitter) if any(jitter) else None
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=True)

    def train(self, mode=True):
        # teacher stays in eval, BatchNorm stats come from the EMA
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update(self, student):
        """One EMA step over parameters and buffers."""
        # warmup, else the random init lingers for thousands of steps
        n = int(self._step.item())
        d = min(self.decay, (1 + n) / (10 + n))
        self._step += 1
        # non-float entries (integer grid sizes, BN step counter) are copied
        pairs = list(zip(self.teacher.parameters(), student.parameters()))
        pairs += list(zip(self.teacher.buffers(), student.buffers()))
        for t, s in pairs:
            if t.is_floating_point() and t.dtype == s.dtype:
                t.lerp_(s.detach(), 1 - d)
            else:
                t.copy_(s)

    def weight(self, global_step):
        """Linear ramp of the consistency weight."""
        return min(1.0, (global_step + 1) / max(self.ramp_steps, 1))

    def jittered(self, color_img):
        """Teacher view: same geometry, independent jitter drawn per window."""
        if self.jitter is None:
            return color_img
        return torch.stack([self.jitter(w) for w in color_img])

    @torch.no_grad()
    def predict(self, color_img, *args, **kwargs):
        return self.teacher(self.jittered(color_img), *args, **kwargs)[0]


def consistency(student_map, teacher_map, channel=0):
    """MSE between the two maps over every cell, mu channel only (nu is unsupervised)."""
    return torch.nn.functional.mse_loss(
        student_map[:, channel], teacher_map[:, channel].detach())
