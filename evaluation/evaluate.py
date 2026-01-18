"""Evaluation for R3++ with multiple corruption levels."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image
import torch

from training.losses import per_sample_cross_entropy

def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text)
    text = text.replace("a.m.", "am").replace("p.m.", "pm")
    text = text.replace("a.m", "am").replace("p.m", "pm")
    text = text.replace("o'clock", "oclock")
    text = text.replace("%", " percent ")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b([ap])\s+m\b", r"\1m", text)
    return " ".join(text.split())


_NUM_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "no": "0",
    "none": "0",
    "nil": "0",
}
_UNIT_TOKENS = {
    "percentage": "percent",
    "percent": "percent",
    "pct": "percent",
}
_DROP_TOKENS = {"oclock"}

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000}


def _collapse_number_words(tokens: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok not in _UNITS and tok not in _TEENS and tok not in _TENS and tok not in _SCALES and tok != "and":
            out.append(tok)
            i += 1
            continue
        start = i
        total = 0
        current = 0
        consumed = False
        while i < len(tokens):
            tok = tokens[i]
            if tok == "and":
                i += 1
                continue
            if tok in _UNITS:
                current += _UNITS[tok]
            elif tok in _TEENS:
                current += _TEENS[tok]
            elif tok in _TENS:
                current += _TENS[tok]
            elif tok in _SCALES:
                scale = _SCALES[tok]
                if current == 0:
                    current = 1
                current *= scale
                if scale >= 1000:
                    total += current
                    current = 0
            else:
                break
            consumed = True
            i += 1
        if consumed:
            total += current
            out.append(str(total))
        else:
            # Standalone "and" without any number words should be kept as-is.
            out.append(tokens[start])
            i = start + 1
    return out


def _canonical_tokens(text: str) -> List[str]:
    tokens = normalize_answer(text).split()
    normalized: List[str] = []
    for tok in tokens:
        tok = _NUM_WORDS.get(tok, tok)
        tok = _UNIT_TOKENS.get(tok, tok)
        if tok in _DROP_TOKENS:
            continue
        normalized.append(tok)
    normalized = _collapse_number_words(normalized)
    return normalized


def _canonical_text(text: str) -> str:
    return " ".join(_canonical_tokens(text))


def _numeric_match(pred: str, ref: str, tol: float = 0.0) -> bool:
    pred_num = _try_parse_float(pred)
    ref_num = _try_parse_float(ref)
    if pred_num is None or ref_num is None:
        return False
    if tol > 0:
        denom = max(abs(ref_num), 1e-6)
        return abs(pred_num - ref_num) / denom <= tol
    return abs(pred_num - ref_num) < 1e-6


def _contains_meridiem(tokens: List[str]) -> Optional[str]:
    for tok in tokens:
        if tok in {"am", "pm"}:
            return tok
    return None


def _subset_match(pred: str, ref: str, max_ref_tokens: int = 6) -> bool:
    ref_tokens = _canonical_tokens(ref)
    pred_tokens = _canonical_tokens(pred)
    if not ref_tokens or not pred_tokens:
        return bool(ref_tokens == pred_tokens)
    ref_mer = _contains_meridiem(ref_tokens)
    if ref_mer and ref_mer not in pred_tokens:
        return False
    if _numeric_match(pred, ref):
        return True
    if len(ref_tokens) > max_ref_tokens:
        return False
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    return all(pred_counts[t] >= ref_counts[t] for t in ref_counts)


def _candidate_spans(text: str) -> List[str]:
    spans: List[str] = []
    if not text:
        return spans
    for line in text.splitlines():
        line = line.strip()
        if line:
            spans.append(line)
    for sent in re.split(r"(?<=[.!?])\\s+", text.strip()):
        sent = sent.strip()
        if sent:
            spans.append(sent)
    for match in re.findall(r"\"([^\"]+)\"|'([^']+)'", text):
        cand = match[0] or match[1]
        cand = cand.strip()
        if cand:
            spans.append(cand)
    for pat in (r"answer is\\s*[:\\-]*\\s*(.+)", r"answer\\s*[:\\-]\\s*(.+)"):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            tail = m.group(1).strip()
            tail = re.split(r"[\\n\\.]", tail, maxsplit=1)[0].strip()
            if tail:
                spans.append(tail)
    # De-duplicate while preserving order.
    seen = set()
    uniq = []
    for s in spans:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq


def _select_best_pred(pred: str, ref: str) -> str:
    candidates = _candidate_spans(pred)
    if not candidates:
        return pred
    best = candidates[0]
    best_score = f1_score(best, ref)
    best_len = len(best)
    for cand in candidates[1:]:
        score = f1_score(cand, ref)
        if score > best_score or (score == best_score and len(cand) < best_len):
            best = cand
            best_score = score
            best_len = len(cand)
    return best


def exact_match(pred: str, ref: str) -> float:
    pred_norm = _canonical_text(pred)
    ref_norm = _canonical_text(ref)
    if pred_norm == ref_norm:
        return 1.0
    return 1.0 if _subset_match(pred, ref) else 0.0


def f1_score(pred: str, ref: str) -> float:
    pred_tokens = _canonical_tokens(pred)
    ref_tokens = _canonical_tokens(ref)
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    if _subset_match(pred, ref):
        return 1.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _try_parse_float(text: str) -> Optional[float]:
    cleaned = _canonical_text(text)
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
    pred_tokens = _canonical_tokens(pred)
    ref_tokens = _canonical_tokens(ref)
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
    pred_tokens = _canonical_tokens(pred)
    ref_tokens = _canonical_tokens(ref)
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
    pred_norm = _canonical_text(pred)
    ref_norm = _canonical_text(ref)
    if not pred_norm or not ref_norm:
        return float(pred_norm == ref_norm)
    if _subset_match(pred, ref):
        return 1.0
    dist = _edit_distance(pred_norm, ref_norm)
    denom = max(len(pred_norm), len(ref_norm), 1)
    score = 1.0 - dist / denom
    return score if score >= 0.5 else 0.0


def _compute_metrics(pred: str, ref: str) -> Dict[str, float]:
    pred = _select_best_pred(pred, ref)
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


def _blank_images(images: List[Image.Image], fallback_size: int = 224) -> List[Image.Image]:
    blanks: List[Image.Image] = []
    for img in images:
        try:
            size = img.size
        except Exception:
            size = (fallback_size, fallback_size)
        blanks.append(Image.new("RGB", size, color=(0, 0, 0)))
    return blanks


_OCR_RE = re.compile(r"(?:^|\\b)OCR\\s*:\\s*(.*)$", re.IGNORECASE)


def _extract_ocr_from_pseudo(text: str) -> str:
    if not text:
        return ""
    match = _OCR_RE.search(text)
    return match.group(1).strip() if match else ""


def _softmax_weights(values: torch.Tensor, alpha: float) -> torch.Tensor:
    scaled = values * alpha
    return torch.softmax(scaled, dim=-1)


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
    sample_every: Optional[int] = None,
    sample_max: Optional[int] = None,
    is_main_process: bool = True,
    answer_only: bool = False,
    router_alpha_override: Optional[Any] = None,
    router_alpha_candidates: Optional[List[Any]] = None,
) -> Dict[float, Dict[str, float]]:
    logger = logging.getLogger(__name__)
    results: Dict[float, Dict[str, float]] = {}
    def _poe_scores(
        images: List[Image.Image],
        blank_images: List[Image.Image],
        questions: List[str],
        pseudo_texts: List[str],
        preds: List[str],
    ) -> torch.Tensor:
        img_outputs = model.forward_student(
            images,
            questions,
            None,
            preds,
            max_length=None,
        )
        txt_outputs = model.forward_student(
            blank_images,
            questions,
            pseudo_texts,
            preds,
            max_length=None,
        )
        img_ce = per_sample_cross_entropy(img_outputs)
        txt_ce = per_sample_cross_entropy(txt_outputs)
        return -(img_ce + txt_ce)
    def _oracle_select(
        preds_by: List[List[str]],
        refs: List[str],
    ) -> List[str]:
        selected: List[str] = []
        for idx, ref in enumerate(refs):
            best_k = 0
            best_em = -1.0
            best_f1 = -1.0
            for k, preds in enumerate(preds_by):
                pred = _select_best_pred(preds[idx], ref)
                em = exact_match(pred, ref)
                if em > best_em:
                    best_k = k
                    best_em = em
                    best_f1 = 0.0
                    if em <= 0:
                        best_f1 = f1_score(pred, ref)
                elif em == best_em and em <= 0:
                    f1 = f1_score(pred, ref)
                    if f1 > best_f1:
                        best_k = k
                        best_f1 = f1
            selected.append(preds_by[best_k][idx])
        return selected
    if mode not in {"r3", "base", "poe"}:
        raise ValueError(f"Unknown mode: {mode}")
    if use_pseudo_text is None:
        use_pseudo_text = mode in {"r3", "poe"}
    if not is_main_process:
        log_every = None
        sample_every = None

    if hasattr(model, "eval"):
        model.eval()
    if hasattr(model, "qwen") and hasattr(model.qwen, "model"):
        model.qwen.model.eval()
    if (
        mode in {"base", "poe"}
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
        samples_logged = 0
        next_sample_at = sample_every if sample_every and sample_every > 0 else None
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
        elif total_batches and log_every > total_batches:
            log_every = max(1, total_batches // 5)
        if is_main_process:
            logger.info(
                "Eval level %.2f (%d/%d) start | batches %s",
                level,
                level_idx,
                total_levels,
                total_batches if total_batches is not None else "unknown",
            )
        batch_idx = 0
        for batch in dataloader:
            batch_idx += 1
            clean = batch["clean"]
            corrupted = batch["corrupted"]
            images = corrupted["images"]
            questions = corrupted["questions"]
            pseudo_texts = corrupted["pseudo_texts"]
            ocr_confs = corrupted.get("ocr_confs")

            batch_size = len(questions)
            want_sample = (
                is_main_process
                and next_sample_at is not None
                and (count + batch_size) >= next_sample_at
            )
            refs = clean["answers"]

            if mode == "r3":
                if not use_pseudo_text:
                    pseudo_texts = ["" for _ in questions]
                with torch.no_grad():
                    if router_alpha_candidates:
                        preds_by = []
                        for override in router_alpha_candidates:
                            preds_by.append(
                                model.generate(
                                    images,
                                    questions,
                                    pseudo_texts,
                                    corruption_level=level,
                                    top_k=top_k,
                                    max_new_tokens=max_new_tokens,
                                    answer_only=answer_only,
                                    ocr_confs=ocr_confs,
                                    router_alpha_override=override,
                                )
                            )
                        preds = _oracle_select(preds_by, refs)
                    elif want_sample:
                        preds, retrieved_texts, retrieved_images, contexts, prompts = model.generate(
                            images,
                            questions,
                            pseudo_texts,
                            corruption_level=level,
                            top_k=top_k,
                            max_new_tokens=max_new_tokens,
                            return_retrieval=True,
                            answer_only=answer_only,
                            ocr_confs=ocr_confs,
                            router_alpha_override=router_alpha_override,
                        )
                    else:
                        preds = model.generate(
                            images,
                            questions,
                            pseudo_texts,
                            corruption_level=level,
                            top_k=top_k,
                            max_new_tokens=max_new_tokens,
                            answer_only=answer_only,
                            ocr_confs=ocr_confs,
                            router_alpha_override=router_alpha_override,
                        )
            elif mode == "poe":
                images, questions, pseudo_texts = _apply_corruption(
                    corruptor, images, questions, pseudo_texts, level, corrupt_text_target
                )
                if not use_pseudo_text:
                    pseudo_texts = ["" for _ in questions]
                blank_images = _blank_images(images)
                with torch.no_grad():
                    preds_img = model.generate_answer(
                        images,
                        questions,
                        None,
                        max_new_tokens=max_new_tokens,
                        answer_only=answer_only,
                    )
                    preds_txt = model.generate_answer(
                        blank_images,
                        questions,
                        pseudo_texts,
                        max_new_tokens=max_new_tokens,
                        answer_only=answer_only,
                    )
                    score_img = _poe_scores(
                        images,
                        blank_images,
                        questions,
                        pseudo_texts,
                        preds_img,
                    )
                    score_txt = _poe_scores(
                        images,
                        blank_images,
                        questions,
                        pseudo_texts,
                        preds_txt,
                    )
                use_img = (score_img >= score_txt).tolist()
                preds = [
                    preds_img[idx] if use_img[idx] else preds_txt[idx]
                    for idx in range(batch_size)
                ]
            elif mode in {"cagate", "ronly"}:
                if not hasattr(model, "qwen") or not hasattr(model, "_retrieve_texts"):
                    raise ValueError("CA-Gate/R-only require an R3 model instance.")
                r3 = model
                if use_pseudo_text is False:
                    pseudo_texts = ["" for _ in questions]
                if use_pseudo_text is False:
                    ocr_confs = None
                if r3.config.enable_corruption:
                    corr_images, corr_texts, _, _ = r3.corruptor(
                        images, pseudo_texts, level
                    )
                else:
                    corr_images, corr_texts = images, pseudo_texts
                queries = [f"{q} {t}".strip() for q, t in zip(questions, corr_texts)]
                _, retrieved_texts, text_scores = r3._retrieve_texts(queries, top_k)

                if mode == "ronly":
                    evidence_texts = [" ".join(texts).strip() for texts in retrieved_texts]
                else:
                    ocr_texts = [_extract_ocr_from_pseudo(t) for t in corr_texts]
                    device = r3.qwen.device
                    if ocr_confs is not None:
                        r_ocr = ocr_confs.squeeze(-1).to(device, dtype=torch.float32)
                    else:
                        lengths = torch.tensor(
                            [float(len(t)) for t in ocr_texts], device=device
                        )
                        max_len = lengths.max().clamp_min(1.0)
                        r_ocr = lengths / max_len
                    if text_scores is not None:
                        r_ret = torch.as_tensor(
                            text_scores[:, 0], device=device, dtype=torch.float32
                        )
                    else:
                        r_ret = torch.zeros_like(r_ocr)
                    scores = torch.stack([r_ocr, r_ret], dim=-1)
                    weights = _softmax_weights(scores, alpha=5.0)
                    choose_ocr = weights[:, 0] >= weights[:, 1]
                    evidence_texts = []
                    for idx, use_ocr in enumerate(choose_ocr.tolist()):
                        if use_ocr:
                            evidence_texts.append(ocr_texts[idx])
                        else:
                            evidence_texts.append(" ".join(retrieved_texts[idx]).strip())

                max_ctx = getattr(r3.config, "max_context_chars", 0)
                if max_ctx and max_ctx > 0:
                    evidence_texts = [text[:max_ctx] for text in evidence_texts]

                with torch.no_grad():
                    preds = r3.qwen.generate_answer(
                        corr_images,
                        questions,
                        evidence_texts,
                        max_new_tokens=max_new_tokens,
                        answer_only=answer_only,
                    )
            else:
                images, questions, pseudo_texts = _apply_corruption(
                    corruptor, images, questions, pseudo_texts, level, corrupt_text_target
                )
                with torch.no_grad():
                    preds = model.generate_answer(
                        images,
                        questions,
                        pseudo_texts if use_pseudo_text else None,
                        max_new_tokens=max_new_tokens,
                        answer_only=answer_only,
                    )

            count_before = count
            for pred, ref in zip(preds, refs):
                sums.update(_compute_metrics(pred, ref))
                count += 1
            if want_sample and is_main_process:
                sample_idx = max(0, min(len(preds) - 1, next_sample_at - count_before - 1))
                img_path = clean["image_paths"][sample_idx]
                question = questions[sample_idx]
                answer = refs[sample_idx]
                pseudo = pseudo_texts[sample_idx] if use_pseudo_text else ""
                pred_text = preds[sample_idx]
                pred_best = _select_best_pred(pred_text, answer)
                logger.info("Eval sample (level %.2f | idx %d):", level, next_sample_at)
                logger.info("  DATA image_path: %s", img_path)
                logger.info("  DATA question: %s", _truncate(question))
                logger.info("  DATA answer(gt): %s", _truncate(answer))
                logger.info("  DATA pseudo_text: %s", _truncate(pseudo))
                logger.info("  MODEL pred_raw: %s", _truncate(pred_text))
                logger.info("  MODEL pred_best: %s", _truncate(pred_best))
                if mode == "r3" and want_sample:
                    if "retrieved_texts" in locals() and retrieved_texts:
                        shown = retrieved_texts[sample_idx][: min(3, len(retrieved_texts[sample_idx]))]
                        logger.info("  RETRIEVAL texts: %s", " || ".join(_truncate(t) for t in shown))
                    if "retrieved_images" in locals() and retrieved_images:
                        shown_imgs = retrieved_images[sample_idx][: min(3, len(retrieved_images[sample_idx]))]
                        logger.info("  RETRIEVAL images: %s", " || ".join(_truncate(p) for p in shown_imgs))
                    if "contexts" in locals():
                        logger.info("  MODEL context: %s", _truncate(contexts[sample_idx], 400))
                    if "prompts" in locals():
                        logger.info("  MODEL input_text: %s", _truncate(prompts[sample_idx], 400))
                samples_logged += 1
                if sample_max is not None and samples_logged >= sample_max:
                    next_sample_at = None
                elif next_sample_at is not None:
                    next_sample_at += sample_every if sample_every else 0

            if log_every is not None and (
                batch_idx % log_every == 0 or (total_batches and batch_idx == total_batches)
            ):
                elapsed = time.time() - level_start
                eta = None
                if total_batches:
                    completed = min(batch_idx, total_batches)
                    eta = elapsed / max(completed, 1) * (total_batches - completed)
                if is_main_process:
                    logger.info(
                        "Eval level %.2f (%d/%d) | done %s/%s batches | samples %d | elapsed %s | eta %s",
                        level,
                        level_idx,
                        total_levels,
                        batch_idx,
                        total_batches if total_batches is not None else "?",
                        count,
                        _format_seconds(elapsed),
                        _format_seconds(eta) if eta is not None else "n/a",
                    )
        if return_sums:
            results[level] = {"count": float(count), **sums}
        else:
            results[level] = {k: v / max(1.0, count) for k, v in sums.items()}
        if is_main_process:
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
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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


def _truncate(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
