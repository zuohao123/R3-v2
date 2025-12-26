"""Evaluation for R3++ with multiple corruption levels."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple



def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, ref: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(ref))


def f1_score(pred: str, ref: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    ref_tokens = normalize_answer(ref).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _try_parse_float(text: str) -> Optional[float]:
    cleaned = normalize_answer(text)
    cleaned = cleaned.replace(",", "")
    match = re.search(r"-?\\d+(?:\\.\\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def relaxed_accuracy(pred: str, ref: str, tol: float = 0.05) -> float:
    """Relaxed accuracy for numeric answers, else exact match."""
    pred_num = _try_parse_float(pred)
    ref_num = _try_parse_float(ref)
    if pred_num is not None and ref_num is not None:
        denom = max(abs(ref_num), 1e-6)
        return float(abs(pred_num - ref_num) / denom <= tol)
    return exact_match(pred, ref)


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(pred: str, ref: str, max_n: int = 4) -> float:
    pred_tokens = normalize_answer(pred).split()
    ref_tokens = normalize_answer(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_counts = _ngram_counts(pred_tokens, n)
        ref_counts = _ngram_counts(ref_tokens, n)
        overlap = sum((pred_counts & ref_counts).values())
        total = max(1, sum(pred_counts.values()))
        precisions.append(overlap / total)
    geo_mean = math.exp(sum(math.log(p + 1e-8) for p in precisions) / max_n)
    bp = math.exp(1 - len(ref_tokens) / len(pred_tokens)) if len(pred_tokens) < len(ref_tokens) else 1.0
    return bp * geo_mean


def _lcs_len(a: List[str], b: List[str]) -> int:
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, token_a in enumerate(a, start=1):
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def rouge_l(pred: str, ref: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    ref_tokens = normalize_answer(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, ch_b in enumerate(b, start=1):
            cur = dp[j]
            if ch_a == ch_b:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = cur
    return dp[-1]


def anls_score(pred: str, ref: str) -> float:
    """ANLS metric used in TextVQA/InfoVQA."""
    pred_norm = normalize_answer(pred)
    ref_norm = normalize_answer(ref)
    if not pred_norm or not ref_norm:
        return float(pred_norm == ref_norm)
    dist = _edit_distance(pred_norm, ref_norm)
    denom = max(len(pred_norm), len(ref_norm), 1)
    score = 1.0 - dist / denom
    return score if score >= 0.5 else 0.0


def _compute_metrics(pred: str, ref: str) -> Dict[str, float]:
    return {
        "exact_match": exact_match(pred, ref),
        "f1": f1_score(pred, ref),
        "bleu": bleu_score(pred, ref),
        "rouge_l": rouge_l(pred, ref),
        "anls": anls_score(pred, ref),
        "relaxed_acc": relaxed_accuracy(pred, ref),
    }


def _apply_corruption(
    corruptor,
    images,
    questions: List[str],
    pseudo_texts: List[str],
    level: float,
    text_target: str,
) -> Tuple[List, List[str], List[str]]:
    if level <= 0.0 or corruptor is None:
        return images, questions, pseudo_texts
    if text_target == "question":
        corr_images, corr_questions, _, _ = corruptor(images, questions, level)
        return corr_images, corr_questions, pseudo_texts
    if text_target == "pseudo_text":
        corr_images, corr_pseudo, _, _ = corruptor(images, pseudo_texts, level)
        return corr_images, questions, corr_pseudo
    corr_images, _, _, _ = corruptor(images, [""] * len(images), level)
    return corr_images, questions, pseudo_texts


def evaluate_model(
    model,
    dataloader,
    corruption_levels: List[float],
    max_new_tokens: int,
    top_k: int,
    return_sums: bool = False,
    mode: str = "r3",
    use_pseudo_text: Optional[bool] = None,
    corrupt_text_target: str = "pseudo_text",
    corruptor=None,
    log_every: Optional[int] = None,
) -> Dict[float, Dict[str, float]]:
    logger = logging.getLogger(__name__)
    results: Dict[float, Dict[str, float]] = {}
    if mode not in {"r3", "base"}:
        raise ValueError(f"Unknown mode: {mode}")
    if use_pseudo_text is None:
        use_pseudo_text = mode == "r3"

    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "qwen") and hasattr(model.qwen, "model"):
        model.qwen.model.eval()
    if (
        mode == "base"
        and corruptor is None
        and any(level > 0 for level in corruption_levels)
    ):
        from config.train_config import R3Config
        from models.r3_modules import CorruptionSimulator

        corruptor = CorruptionSimulator(R3Config())

    total_levels = len(corruption_levels)
    for level_idx, level in enumerate(corruption_levels, start=1):
        sums = Counter()
        count = 0
        try:
            total_batches = len(dataloader)
        except TypeError:
            total_batches = None
        level_start = time.time()
        if log_every is None:
            if total_batches:
                log_every = max(1, total_batches // 20)
            else:
                log_every = 50
        logger.info(
            "Eval level %.2f (%d/%d) start | batches %s",
            level,
            level_idx,
            total_levels,
            total_batches if total_batches is not None else "unknown",
        )
        for batch in dataloader:
            clean = batch["clean"]
            corrupted = batch["corrupted"]
            images = corrupted["images"]
            questions = corrupted["questions"]
            pseudo_texts = corrupted["pseudo_texts"]

            if mode == "r3":
                if not use_pseudo_text:
                    pseudo_texts = ["" for _ in questions]
                preds = model.generate(
                    images,
                    questions,
                    pseudo_texts,
                    corruption_level=level,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                )
            else:
                images, questions, pseudo_texts = _apply_corruption(
                    corruptor, images, questions, pseudo_texts, level, corrupt_text_target
                )
                preds = model.generate_answer(
                    images,
                    questions,
                    pseudo_texts if use_pseudo_text else None,
                    max_new_tokens=max_new_tokens,
                )

            refs = clean["answers"]
            for pred, ref in zip(preds, refs):
                sums.update(_compute_metrics(pred, ref))
                count += 1
            if count % log_every == 0:
                elapsed = time.time() - level_start
                eta = None
                if total_batches:
                    completed = min(count, total_batches)
                    eta = elapsed / max(completed, 1) * (total_batches - completed)
                logger.info(
                    "Eval level %.2f (%d/%d) | done %s/%s | elapsed %s | eta %s",
                    level,
                    level_idx,
                    total_levels,
                    count,
                    total_batches if total_batches is not None else "?",
                    _format_seconds(elapsed),
                    _format_seconds(eta) if eta is not None else "n/a",
                )
        if return_sums:
            results[level] = {"count": float(count), **sums}
        else:
            results[level] = {k: v / max(1.0, count) for k, v in sums.items()}
        logger.info(
            "Eval level %.2f (%d/%d) done | samples %d | elapsed %s",
            level,
            level_idx,
            total_levels,
            count,
            _format_seconds(time.time() - level_start),
        )
    return results


def save_results(results: Dict[float, Dict[str, float]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}h{mins:02d}m{secs:02d}s"
    return f"{mins:02d}m{secs:02d}s"
