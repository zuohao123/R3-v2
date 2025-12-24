"""Evaluation for R3++ with multiple corruption levels."""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Dict, Iterable, List



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


def evaluate_model(
    r3,
    dataloader,
    corruption_levels: List[float],
    max_new_tokens: int,
    top_k: int,
    return_sums: bool = False,
) -> Dict[float, Dict[str, float]]:
    results: Dict[float, Dict[str, float]] = {}
    r3.eval()
    if hasattr(r3, "qwen") and hasattr(r3.qwen, "model"):
        r3.qwen.model.eval()
    for level in corruption_levels:
        sums = Counter()
        count = 0
        for batch in dataloader:
            clean = batch["clean"]
            corrupted = batch["corrupted"]
            preds = r3.generate(
                corrupted["images"],
                corrupted["questions"],
                corrupted["pseudo_texts"],
                corruption_level=level,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )
            refs = clean["answers"]
            for pred, ref in zip(preds, refs):
                sums.update(
                    {
                        "exact_match": exact_match(pred, ref),
                        "f1": f1_score(pred, ref),
                        "bleu": bleu_score(pred, ref),
                        "rouge_l": rouge_l(pred, ref),
                    }
                )
                count += 1
        if return_sums:
            results[level] = {"count": float(count), **sums}
        else:
            results[level] = {k: v / max(1.0, count) for k, v in sums.items()}
    return results


def save_results(results: Dict[float, Dict[str, float]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
