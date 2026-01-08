"""Corruption curriculum scheduler."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumScheduler:
    """Corruption schedule controller."""
    max_corruption: float
    warmup_steps: int
    total_steps: int
    schedule: str = "linear"
    cycles: int = 1

    def get_level(self, step: int) -> float:
        if self.total_steps <= 0:
            return self.max_corruption
        step = max(step, 0)
        if self.schedule == "cyclic":
            cycles = max(1, int(self.cycles))
            cycle_len = max(1, self.total_steps // cycles)
            cycle_step = step % cycle_len
            return self.max_corruption * (cycle_step / cycle_len)
        if step <= 0:
            return 0.0
        if step < self.warmup_steps:
            return self.max_corruption * (step / max(1, self.warmup_steps))
        if step >= self.total_steps:
            return self.max_corruption
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return min(self.max_corruption, self.max_corruption * progress)
