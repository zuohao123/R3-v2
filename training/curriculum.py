"""Corruption curriculum scheduler."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumScheduler:
    """Linear warmup to maximum corruption."""
    max_corruption: float
    warmup_steps: int
    total_steps: int

    def get_level(self, step: int) -> float:
        if step <= 0:
            return 0.0
        if step < self.warmup_steps:
            return self.max_corruption * (step / max(1, self.warmup_steps))
        if step >= self.total_steps:
            return self.max_corruption
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return min(self.max_corruption, self.max_corruption * progress)
