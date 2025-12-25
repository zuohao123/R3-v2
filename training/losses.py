"""Losses for R3++ training."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def _safe_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    vocab = logits.size(-1)
    loss = F.cross_entropy(
        logits.view(-1, vocab),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    loss = loss.view(labels.size())
    mask = labels.ne(ignore_index)
    denom = mask.sum(dim=1).clamp_min(1).float()
    per_sample = loss.sum(dim=1) / denom
    return per_sample.mean()


def _label_ratio(labels: torch.Tensor, ignore_index: int = -100) -> float:
    mask = labels.ne(ignore_index)
    return mask.float().mean().item() if labels.numel() > 0 else 0.0


def cross_entropy_loss(outputs: Any) -> torch.Tensor:
    """Return a stable loss value, falling back to custom CE when labels exist."""
    if hasattr(outputs, "labels") and outputs.labels is not None and hasattr(outputs, "logits"):
        return _safe_cross_entropy(outputs.logits, outputs.labels)
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
    metrics: Dict[str, torch.Tensor] = {"total": total, "ce": ce, "consistency": cons}
    if hasattr(student_outputs, "labels") and student_outputs.labels is not None:
        ratio = _label_ratio(student_outputs.labels)
        metrics["label_ratio"] = torch.tensor(ratio, device=ce.device)
    return metrics
