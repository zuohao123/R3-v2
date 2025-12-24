"""Losses for R3++ training."""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F


def cross_entropy_loss(outputs: Any) -> torch.Tensor:
    """Return model loss if available."""
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss
    raise ValueError("Model outputs do not include a loss value.")


def consistency_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """KL divergence between student and teacher logits."""
    student = F.log_softmax(student_logits / temperature, dim=-1)
    teacher = F.softmax(teacher_logits / temperature, dim=-1)
    min_len = min(student.size(1), teacher.size(1))
    student = student[:, :min_len, :]
    teacher = teacher[:, :min_len, :]
    return F.kl_div(student, teacher, reduction="batchmean") * (temperature**2)


def compute_total_loss(
    student_outputs: Any,
    teacher_outputs: Any,
    consistency_weight: float,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    """Compute total loss dict for logging."""
    ce = cross_entropy_loss(student_outputs)
    if teacher_outputs is None or consistency_weight <= 0:
        cons = torch.tensor(0.0, device=ce.device)
        total = ce
    else:
        cons = consistency_loss(student_outputs.logits, teacher_outputs.logits, temperature)
        total = ce + consistency_weight * cons
    return {"total": total, "ce": ce, "consistency": cons}
