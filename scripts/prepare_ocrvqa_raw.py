#!/usr/bin/env python3
"""Download OCR-VQA and prepare raw JSONL + images."""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image


def _ensure_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, list):
        value = value[0] if value else fallback
    if isinstance(value, dict):
        value = value.get("text", value)
    text = str(value).strip()
    return text if text else fallback


def _extract_answer(raw: Dict[str, Any], fallback: str) -> str:
    for key in (
        "answer",
        "answers",
        "label",
        "gt_answer",
        "answer_text",
        "full_answer",
        "ground_truth",
        "text",
    ):
        if key in raw and raw[key] not in (None, ""):
            return _ensure_text(raw[key], fallback)
    return fallback


def _get_first(sample: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in sample and sample[key] not in (None, ""):
            return sample[key]
    return None


def _safe_name(text: str) -> str:
    if not text:
        return "unknown"
    text = re.sub(r"[^0-9a-zA-Z_-]+", "_", text.strip())
    return text.strip("_") or "unknown"


def _load_image(payload: Any) -> Optional[Image.Image]:
    if isinstance(payload, Image.Image):
        return payload.convert("RGB")
    if isinstance(payload, dict):
        data = payload.get("bytes")
        if data:
            return Image.open(io.BytesIO(data)).convert("RGB")
        path = payload.get("path")
        if path and os.path.exists(path):
            return Image.open(path).convert("RGB")
    if isinstance(payload, str):
        try:
            b64 = payload.split(",", 1)[1] if "," in payload else payload
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            if os.path.exists(payload):
                return Image.open(payload).convert("RGB")
    return None


def _pick_split(dataset_name: str, config_name: Optional[str], split: Optional[str]) -> str:
    from datasets import get_dataset_split_names

    if split:
        return split
    splits = (
        get_dataset_split_names(dataset_name, config_name)
        if config_name
        else get_dataset_split_names(dataset_name)
    )
    for cand in ("test", "validation", "val", "dev", "train"):
        if cand in splits:
            return cand
    if not splits:
        raise ValueError(f"No splits found for dataset: {dataset_name}")
    return splits[0]


def _iter_dataset(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    streaming: bool,
    shuffle: bool,
    seed: int,
) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    if config_name:
        ds = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)
    if not streaming and shuffle:
        ds = ds.shuffle(seed=seed)
    return ds


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OCR-VQA raw JSONL and images.")
    parser.add_argument("--dataset", default="howard-hou/OCR-VQA")
    parser.add_argument("--config", default=None, help="Optional HF config name.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--out_dir", default="data/raw/ocrvqa")
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--image_subdir", default="images")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--out_jsonl", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    split = _pick_split(args.dataset, args.config, args.split)
    images_dir = os.path.join(args.out_dir, args.image_subdir)
    os.makedirs(images_dir, exist_ok=True)

    out_jsonl = args.out_jsonl or os.path.join(args.out_dir, f"ocrvqa_raw_{split}.jsonl")
    kept = 0
    total = 0
    records: List[Dict[str, Any]] = []

    for sample in _iter_dataset(
        args.dataset, args.config, split, args.streaming, args.shuffle, args.seed
    ):
        total += 1
        question = _ensure_text(
            _get_first(sample, ["question", "query", "text", "caption", "prompt"]),
            "",
        )
        answer = _extract_answer(sample, "")
        if not question or not answer:
            continue
        image_value = _get_first(
            sample, ["image", "img", "image_path", "image_file", "image_filename"]
        )
        image = _load_image(image_value)
        if image is None:
            continue
        image_id = _ensure_text(
            _get_first(
                sample, ["image_id", "image_uid", "img_id", "image_name", "file_name", "id"]
            ),
            f"{kept:06d}",
        )
        filename = f"{split}_{_safe_name(image_id)}_{kept:06d}.png"
        abs_path = os.path.join(images_dir, filename)
        image.save(abs_path)
        rel_path = os.path.relpath(abs_path, args.image_root).replace(os.path.sep, "/")
        records.append(
            {
                "image_path": rel_path,
                "question": question,
                "answer": answer,
                "split": split,
            }
        )
        kept += 1
        if args.max_samples and kept >= args.max_samples:
            break
        if kept % 500 == 0:
            logging.info("Collected %d samples (seen %d).", kept, total)

    _write_jsonl(out_jsonl, records)
    logging.info("Saved %d samples to %s (split=%s).", kept, out_jsonl, split)


if __name__ == "__main__":
    main()
