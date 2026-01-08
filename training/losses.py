"""Losses for R3++ training."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from config.train_config import LossConfig

def _per_sample_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    logits = logits.float()
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    logits = logits.clamp(-50.0, 50.0)
    if logits.size(1) < 2 or labels.size(1) < 2:
        return torch.zeros((logits.size(0),), device=logits.device)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.size(-1)
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    loss = loss.view(shift_labels.size())
    mask = shift_labels.ne(ignore_index)
    denom = mask.sum(dim=1).clamp_min(1).float()
    per_sample = loss.sum(dim=1) / denom
    return per_sample


def _safe_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    per_sample = _per_sample_cross_entropy(logits, labels, ignore_index=ignore_index)
    return per_sample.mean()


def _label_ratio(labels: torch.Tensor, ignore_index: int = -100) -> float:
    if labels.size(1) < 2:
        return 0.0
    mask = labels[:, 1:].ne(ignore_index)
    return mask.float().mean().item() if labels.numel() > 0 else 0.0


def cross_entropy_loss(outputs: Any) -> torch.Tensor:
    """Return a stable loss value, falling back to custom CE when labels exist."""
    if hasattr(outputs, "labels") and outputs.labels is not None and hasattr(outputs, "logits"):
        return _safe_cross_entropy(outputs.logits, outputs.labels)
    raise ValueError("Model outputs do not include a loss value.")


def per_sample_cross_entropy(outputs: Any) -> torch.Tensor:
    """Return per-sample CE values for router supervision."""
    if hasattr(outputs, "labels") and outputs.labels is not None and hasattr(outputs, "logits"):
        return _per_sample_cross_entropy(outputs.logits, outputs.labels)
    raise ValueError("Model outputs do not include labels/logits for per-sample CE.")


def consistency_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """KL divergence between student and teacher logits."""
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    student_logits = torch.nan_to_num(student_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    teacher_logits = torch.nan_to_num(teacher_logits, nan=0.0, posinf=1e4, neginf=-1e4)
    student = F.log_softmax(student_logits / temperature, dim=-1)
    teacher = F.softmax(teacher_logits / temperature, dim=-1)
    student = torch.nan_to_num(student, nan=0.0, posinf=0.0, neginf=0.0)
    teacher = torch.nan_to_num(teacher, nan=0.0, posinf=0.0, neginf=0.0)
    min_len = min(student.size(1), teacher.size(1))
    student = student[:, :min_len, :]
    teacher = teacher[:, :min_len, :]
    loss = F.kl_div(student, teacher, reduction="batchmean") * (temperature**2)
    denom = max(min_len, 1)
    return loss / float(denom)


def gate_confidence_loss(
    gates: torch.Tensor, c_vis: torch.Tensor, c_text: torch.Tensor
) -> torch.Tensor:
    eps = 1e-6
    conf_t = c_text.clamp(0.0, 1.0)
    conf_v = c_vis.clamp(0.0, 1.0)
    target = torch.cat([1.0 - conf_t, 1.0 - conf_v, conf_v], dim=-1)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    target = target.detach()
    gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(eps)
    gates = gates.clamp_min(eps)
    return F.kl_div(torch.log(gates), target, reduction="batchmean")


def gate_entropy_loss(gates: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(gates * torch.log(gates.clamp_min(eps))).sum(dim=-1)
    return -entropy.mean()


def retrieval_alignment_loss(
    mem_t: torch.Tensor,
    mem_i: torch.Tensor,
    temperature: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    eps = 1e-6
    mem_t = torch.nan_to_num(mem_t, nan=0.0, posinf=0.0, neginf=0.0)
    mem_i = torch.nan_to_num(mem_i, nan=0.0, posinf=0.0, neginf=0.0)
    mem_t = F.normalize(mem_t, dim=-1, eps=eps)
    mem_i = F.normalize(mem_i, dim=-1, eps=eps)
    logits = mem_t @ mem_i.T
    logits = logits / max(temperature, eps)
    log_probs = F.log_softmax(logits, dim=1)
    pos = torch.diag(log_probs)
    loss_t2i = -pos
    log_probs_t = F.log_softmax(logits.T, dim=1)
    pos_t = torch.diag(log_probs_t)
    loss_i2t = -pos_t
    loss = 0.5 * (loss_t2i + loss_i2t)
    if weights is not None:
        weights = weights.float().clamp_min(0.0)
        denom = weights.sum().clamp_min(eps)
        return (loss * weights).sum() / denom
    return loss.mean()


def compute_total_loss(
    student_outputs: Any,
    teacher_outputs: Any,
    consistency_weight: float,
    temperature: float,
    loss_cfg: Optional[LossConfig] = None,
) -> Dict[str, torch.Tensor]:
    """Compute total loss dict for logging."""
    gate_payload: Optional[Dict[str, torch.Tensor]] = None
    outputs = student_outputs
    if isinstance(student_outputs, dict) and "outputs" in student_outputs:
        outputs = student_outputs["outputs"]
        gate_payload = student_outputs

    ce = cross_entropy_loss(outputs)
    if teacher_outputs is None or consistency_weight <= 0:
        cons = torch.tensor(0.0, device=ce.device)
        total = ce
    else:
        cons = consistency_loss(outputs.logits, teacher_outputs.logits, temperature)
        total = ce + consistency_weight * cons
    metrics: Dict[str, torch.Tensor] = {"total": total, "ce": ce, "consistency": cons}
    if loss_cfg is not None and gate_payload is not None:
        gates = gate_payload.get("gates")
        c_vis = gate_payload.get("c_vis")
        c_text = gate_payload.get("c_text")
        if gates is not None and c_vis is not None and c_text is not None:
            if loss_cfg.gate_conf_weight > 0:
                g_conf = gate_confidence_loss(gates, c_vis, c_text)
                metrics["gate_conf"] = g_conf
                total = total + loss_cfg.gate_conf_weight * g_conf
            if loss_cfg.gate_entropy_weight > 0:
                g_ent = gate_entropy_loss(gates)
                metrics["gate_entropy"] = g_ent
                total = total + loss_cfg.gate_entropy_weight * g_ent
            metrics["total"] = total
        if (
            loss_cfg.retrieval_align_weight > 0
            and gate_payload.get("retrieval_ready")
            and gate_payload.get("mem_t") is not None
            and gate_payload.get("mem_i") is not None
        ):
            mem_t = gate_payload["mem_t"]
            mem_i = gate_payload["mem_i"]
            weights = None
            if c_vis is not None and c_text is not None:
                weights = (c_vis * c_text).squeeze(-1)
            r_align = retrieval_alignment_loss(
                mem_t, mem_i, loss_cfg.retrieval_align_temperature, weights=weights
            )
            metrics["retrieval_align"] = r_align
            metrics["total"] = metrics["total"] + loss_cfg.retrieval_align_weight * r_align
    if hasattr(outputs, "labels") and outputs.labels is not None:
        ratio = _label_ratio(outputs.labels)
        metrics["label_ratio"] = torch.tensor(ratio, device=ce.device)
    return metrics
