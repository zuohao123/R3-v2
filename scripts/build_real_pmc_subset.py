#!/usr/bin/env python3
"""Mine a small real-PMC subset using OCR and image heuristics."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _iter_jsonl(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _normalize_for_match(text: str) -> str:
    tokens = _tokenize(text)
    return " ".join(tokens)


def _answer_in_ocr(answer: str, ocr_text: str, max_tokens: int) -> bool:
    if not answer or not ocr_text:
        return False
    answer_norm = _normalize_for_match(answer)
    ocr_norm = _normalize_for_match(ocr_text)
    if answer_norm and answer_norm in ocr_norm:
        return True
    answer_tokens = _tokenize(answer)
    if not answer_tokens or len(answer_tokens) > max_tokens:
        return False
    ocr_tokens = set(_tokenize(ocr_text))
    return all(token in ocr_tokens for token in answer_tokens)


def _match_prefix(path: str, prefix: str) -> bool:
    prefix = prefix.strip().strip("/")
    if not prefix:
        return False
    norm = path.replace("\\", "/").lstrip("./")
    if norm.startswith(prefix + "/"):
        return True
    parts = [p for p in norm.split("/") if p]
    return prefix in parts


def _shard_path(out_jsonl: str, shard_id: int) -> str:
    base, ext = os.path.splitext(out_jsonl)
    if not ext:
        ext = ".jsonl"
    return f"{base}.shard{shard_id}{ext}"


def _downsample(
    candidates: List[Dict[str, str]],
    max_samples: Optional[int],
    max_samples_per_label: Optional[int],
    seed: int,
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    if max_samples_per_label:
        grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for item in candidates:
            grouped[item.get("pmc_label", "unknown")].append(item)
        candidates = []
        for label, items in grouped.items():
            rng.shuffle(items)
            candidates.extend(items[: max_samples_per_label])
    if max_samples and len(candidates) > max_samples:
        rng.shuffle(candidates)
        candidates = candidates[: max_samples]
    return candidates


def _load_ocr_map(path: Optional[str]) -> Dict[str, Dict[str, Optional[float]]]:
    if not path:
        return {}
    ocr_map: Dict[str, Dict[str, Optional[float]]] = {}
    for record in _iter_jsonl(path):
        image_path = record.get("image_path")
        if not image_path:
            continue
        text = record.get("ocr_text") or record.get("text") or ""
        conf = record.get("ocr_conf") or record.get("ocr_conf_mean") or record.get("conf")
        conf_val: Optional[float] = None
        if conf is not None:
            try:
                conf_val = float(conf)
            except (TypeError, ValueError):
                conf_val = None
        ocr_map[image_path] = {"text": str(text or ""), "conf": conf_val}
    return ocr_map


def _resolve_image_path(path: str, image_root: Optional[str]) -> str:
    if image_root and not os.path.isabs(path):
        return os.path.join(image_root, path)
    return path


def _compute_low_var_ratios(
    image: Image.Image, grid: int, threshold: float
) -> Tuple[float, float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for image heuristics") from exc
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    h, w = arr.shape
    step_h = max(1, h // grid)
    step_w = max(1, w // grid)
    low_var = 0
    low_var_border = 0
    total = 0
    border = 0
    for yi in range(0, h, step_h):
        for xi in range(0, w, step_w):
            patch = arr[yi : yi + step_h, xi : xi + step_w]
            var = float(patch.var())
            is_border = yi == 0 or xi == 0 or (yi + step_h) >= h or (xi + step_w) >= w
            total += 1
            if is_border:
                border += 1
            if var <= threshold:
                low_var += 1
                if is_border:
                    low_var_border += 1
    low_var_ratio = low_var / max(1, total)
    border_ratio = low_var_border / max(1, border)
    return low_var_ratio, border_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-PMC eval subset.")
    parser.add_argument("--val_jsonl", default=None, help="Input eval JSONL.")
    parser.add_argument("--out_jsonl", required=True, help="Output subset JSONL.")
    parser.add_argument("--ocr_jsonl", default=None, help="Optional OCR cache JSONL.")
    parser.add_argument("--image_root", default=None, help="Root directory for images.")
    parser.add_argument(
        "--dataset_prefixes",
        default="screenqa,chartqa,infovqa",
        help="Comma-separated prefixes to track stats.",
    )
    parser.add_argument("--min_pseudo_tokens", type=int, default=8)
    parser.add_argument("--min_ocr_tokens", type=int, default=6)
    parser.add_argument("--min_ocr_conf", type=float, default=0.5)
    parser.add_argument("--answer_max_tokens", type=int, default=3)
    parser.add_argument(
        "--max_ocr_ratio",
        type=float,
        default=0.5,
        help="Keep sample if ocr_tokens/pseudo_tokens <= this ratio.",
    )
    parser.add_argument(
        "--max_overlap",
        type=float,
        default=0.3,
        help="Keep sample if token overlap <= this ratio.",
    )
    parser.add_argument("--min_ocr_signals", type=int, default=2)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--low_var_threshold", type=float, default=15.0)
    parser.add_argument("--low_var_ratio", type=float, default=0.35)
    parser.add_argument("--border_low_var_ratio", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_samples_per_label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument(
        "--merge_shards",
        action="store_true",
        help="Merge shard outputs and apply global sampling.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prefixes = [p.strip() for p in args.dataset_prefixes.split(",") if p.strip()]

    if not args.merge_shards and not args.val_jsonl:
        parser.error("--val_jsonl is required unless --merge_shards is set.")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards).")

    if args.merge_shards:
        merged: List[Dict[str, str]] = []
        for shard_id in range(args.num_shards):
            shard_path = _shard_path(args.out_jsonl, shard_id)
            if not os.path.exists(shard_path):
                logging.warning("Missing shard: %s", shard_path)
                continue
            merged.extend(list(_iter_jsonl(shard_path)))
        merged = _downsample(
            merged, args.max_samples, args.max_samples_per_label, args.seed
        )
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for item in merged:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logging.info(
            "Merged %d shards into %s (%d samples).",
            args.num_shards,
            args.out_jsonl,
            len(merged),
        )
        return

    ocr_map = _load_ocr_map(args.ocr_jsonl)
    if args.image_root and Image is None:
        logging.warning("PIL is not available; skipping image heuristics.")

    candidates: List[Dict[str, str]] = []
    stats = defaultdict(Counter)
    for record in _iter_jsonl(args.val_jsonl):
        image_path = record.get("image_path")
        if not image_path:
            continue
        pseudo_text = record.get("pseudo_text", "")
        if len(_tokenize(pseudo_text)) < args.min_pseudo_tokens:
            continue
        ocr_text = record.get("ocr_text") or ""
        ocr_conf = record.get("ocr_conf")
        ocr_cache = ocr_map.get(image_path, {})
        if not ocr_text and ocr_cache:
            ocr_text = ocr_cache.get("text", "") or ""
        if ocr_conf is None and ocr_cache:
            ocr_conf = ocr_cache.get("conf")
        try:
            ocr_conf_val = float(ocr_conf) if ocr_conf is not None else None
        except (TypeError, ValueError):
            ocr_conf_val = None

        ocr_tokens = _tokenize(ocr_text)
        pseudo_tokens = _tokenize(pseudo_text)
        ocr_ratio = len(ocr_tokens) / max(1, len(pseudo_tokens))
        overlap = len(set(ocr_tokens) & set(pseudo_tokens)) / max(
            1, len(set(pseudo_tokens))
        )
        answer = str(record.get("answer", "") or "")
        answer_in_ocr = _answer_in_ocr(answer, ocr_text, args.answer_max_tokens)

        ocr_signals: List[str] = []
        if len(ocr_tokens) <= args.min_ocr_tokens:
            ocr_signals.append("short_ocr")
        if ocr_conf_val is not None and ocr_conf_val < args.min_ocr_conf:
            ocr_signals.append("low_ocr_conf")
        if not answer_in_ocr and answer:
            ocr_signals.append("answer_missing")
        if ocr_ratio <= args.max_ocr_ratio:
            ocr_signals.append("low_ocr_ratio")
        if overlap <= args.max_overlap:
            ocr_signals.append("low_overlap")
        ocr_drop = len(ocr_signals) >= args.min_ocr_signals

        vision_signals: List[str] = []
        low_var_ratio = None
        border_low_var = None
        if args.image_root and Image is not None:
            resolved = _resolve_image_path(image_path, args.image_root)
            try:
                image = Image.open(resolved).convert("RGB")
                low_var_ratio, border_low_var = _compute_low_var_ratios(
                    image, args.grid, args.low_var_threshold
                )
                if low_var_ratio >= args.low_var_ratio:
                    vision_signals.append("low_var_ratio")
                if border_low_var >= args.border_low_var_ratio:
                    vision_signals.append("border_low_var_ratio")
            except (FileNotFoundError, OSError, ValueError, RuntimeError):
                pass
        vision_occlusion = bool(vision_signals)

        if not ocr_drop and not vision_occlusion:
            continue

        if ocr_drop and vision_occlusion:
            label = "mixed"
        elif ocr_drop:
            label = "ocr_dropout"
        else:
            label = "vision_occlusion"

        item = dict(record)
        item["pmc_label"] = label
        item["pmc_ocr_ratio"] = round(ocr_ratio, 4)
        item["pmc_overlap"] = round(overlap, 4)
        item["pmc_ocr_conf"] = None if ocr_conf_val is None else round(ocr_conf_val, 4)
        item["pmc_answer_in_ocr"] = bool(answer_in_ocr)
        item["pmc_ocr_signals"] = ocr_signals
        item["pmc_vision_signals"] = vision_signals
        if low_var_ratio is not None:
            item["pmc_low_var_ratio"] = round(low_var_ratio, 4)
        if border_low_var is not None:
            item["pmc_border_low_var_ratio"] = round(border_low_var, 4)
        candidates.append(item)

        matched = False
        for prefix in prefixes:
            if _match_prefix(image_path, prefix):
                stats[prefix][label] += 1
                matched = True
                break
        if not matched:
            stats["other"][label] += 1

    candidates = _downsample(
        candidates, args.max_samples, args.max_samples_per_label, args.seed
    )
    out_jsonl = args.out_jsonl
    if args.num_shards > 1:
        out_jsonl = _shard_path(args.out_jsonl, args.shard_id)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for item in candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = len(candidates)
    logging.info("Saved %d real-PMC samples to %s", total, out_jsonl)
    for key in sorted(stats.keys()):
        logging.info("%s: %s", key, dict(stats[key]))


if __name__ == "__main__":
    main()
