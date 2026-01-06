#!/usr/bin/env python3
"""Build a small real-PMC subset using OCR mismatch heuristics."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List


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


def _match_prefix(path: str, prefix: str) -> bool:
    prefix = prefix.strip().strip("/")
    if not prefix:
        return False
    norm = path.replace("\\", "/").lstrip("./")
    if norm.startswith(prefix + "/"):
        return True
    parts = [p for p in norm.split("/") if p]
    return prefix in parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-PMC eval subset.")
    parser.add_argument("--val_jsonl", required=True, help="Input eval JSONL.")
    parser.add_argument("--ocr_jsonl", required=True, help="OCR cache JSONL.")
    parser.add_argument("--out_jsonl", required=True, help="Output subset JSONL.")
    parser.add_argument(
        "--dataset_prefixes",
        default="screenqa,chartqa,infovqa",
        help="Comma-separated prefixes to track stats.",
    )
    parser.add_argument("--min_pseudo_tokens", type=int, default=8)
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
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prefixes = [p.strip() for p in args.dataset_prefixes.split(",") if p.strip()]
    ocr_map: Dict[str, str] = {}
    for record in _iter_jsonl(args.ocr_jsonl):
        image_path = record.get("image_path")
        if image_path:
            ocr_map[image_path] = record.get("ocr_text", "")

    candidates: List[Dict[str, str]] = []
    stats = defaultdict(int)
    for record in _iter_jsonl(args.val_jsonl):
        image_path = record.get("image_path")
        if not image_path or image_path not in ocr_map:
            continue
        pseudo_text = record.get("pseudo_text", "")
        ocr_text = ocr_map.get(image_path, "")
        pseudo_tokens = _tokenize(pseudo_text)
        if len(pseudo_tokens) < args.min_pseudo_tokens:
            continue
        ocr_tokens = _tokenize(ocr_text)
        ocr_ratio = len(ocr_tokens) / max(1, len(pseudo_tokens))
        overlap = len(set(ocr_tokens) & set(pseudo_tokens)) / max(
            1, len(set(pseudo_tokens))
        )
        if ocr_ratio > args.max_ocr_ratio and overlap > args.max_overlap:
            continue
        item = dict(record)
        item["pseudo_text"] = ocr_text
        item["pmc_ocr_ratio"] = round(ocr_ratio, 4)
        item["pmc_overlap"] = round(overlap, 4)
        item["pseudo_text_orig"] = pseudo_text
        candidates.append(item)
        matched = False
        for prefix in prefixes:
            if _match_prefix(image_path, prefix):
                stats[prefix] += 1
                matched = True
                break
        if not matched:
            stats["other"] += 1

    if args.max_samples and len(candidates) > args.max_samples:
        random.Random(args.seed).shuffle(candidates)
        candidates = candidates[: args.max_samples]

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for item in candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total = len(candidates)
    print(f"Saved {total} real-PMC samples to {args.out_jsonl}")
    for key in sorted(stats.keys()):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
