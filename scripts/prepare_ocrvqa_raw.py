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
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _ensure_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, list):
        value = value[0] if value else fallback
    if isinstance(value, dict):
        value = value.get("text", value.get("answer", value))
    text = str(value).strip()
    return text if text else fallback


def _extract_answer_value(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, list):
        answers: List[str] = []
        for item in value:
            if isinstance(item, dict):
                for key in ("answer", "text", "label", "value"):
                    if key in item and item[key] not in (None, ""):
                        answers.append(_ensure_text(item[key], ""))
                        break
                else:
                    answers.append(_ensure_text(item, ""))
            else:
                answers.append(_ensure_text(item, ""))
        answers = [a for a in answers if a]
        if not answers:
            return fallback
        norms = [_normalize_answer(a) for a in answers]
        top_norm = Counter(norms).most_common(1)[0][0]
        for a in answers:
            if _normalize_answer(a) == top_norm:
                return a
        return answers[0]
    if isinstance(value, dict):
        for key in ("answer", "text", "label", "value"):
            if key in value and value[key] not in (None, ""):
                return _ensure_text(value[key], fallback)
        return _ensure_text(value, fallback)
    return _ensure_text(value, fallback)


def _extract_answer(raw: Dict[str, Any], fallback: str) -> str:
    keys = (
        "answer",
        "answers",
        "label",
        "gt_answer",
        "answer_text",
        "full_answer",
        "ground_truth",
        "text",
    )
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return _extract_answer_value(raw[key], fallback)
    for key, value in raw.items():
        if "answer" in str(key).lower() and value not in (None, ""):
            return _extract_answer_value(value, fallback)
    return fallback


def _get_first(sample: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in sample and sample[key] not in (None, ""):
            return sample[key]
    return None


def _find_by_terms(sample: Dict[str, Any], terms: Tuple[str, ...]) -> Optional[Any]:
    for key, value in sample.items():
        key_l = str(key).lower()
        if any(term in key_l for term in terms) and value not in (None, ""):
            return value
    return None


def _safe_name(text: str) -> str:
    if not text:
        return "unknown"
    text = re.sub(r"[^0-9a-zA-Z_-]+", "_", text.strip())
    return text.strip("_") or "unknown"


def _load_image(
    payload: Any,
    dataset_name: Optional[str] = None,
    download_images: bool = False,
    repo_prefix: str = "",
) -> Optional[Image.Image]:
    if isinstance(payload, Image.Image):
        return payload.convert("RGB")
    if isinstance(payload, list) and payload:
        payload = payload[0]
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
            if download_images and dataset_name:
                if payload.startswith("http"):
                    try:
                        import requests

                        resp = requests.get(payload, timeout=30)
                        resp.raise_for_status()
                        return Image.open(io.BytesIO(resp.content)).convert("RGB")
                    except Exception:
                        return None
                try:
                    from huggingface_hub import hf_hub_download

                    remote_path = os.path.join(repo_prefix, payload).replace("\\", "/")
                    local_path = hf_hub_download(
                        repo_id=dataset_name,
                        repo_type="dataset",
                        filename=remote_path,
                    )
                    return Image.open(local_path).convert("RGB")
                except Exception:
                    return None
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
    parser.add_argument("--image_key", default=None, help="Explicit image field name.")
    parser.add_argument("--download_images", action="store_true")
    parser.add_argument(
        "--image_repo_prefix",
        default="",
        help="Prefix path inside HF dataset repo for image files.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--out_jsonl", default=None)
    parser.add_argument("--debug_keys", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    split = _pick_split(args.dataset, args.config, args.split)
    images_dir = os.path.join(args.out_dir, args.image_subdir)
    os.makedirs(images_dir, exist_ok=True)

    out_jsonl = args.out_jsonl or os.path.join(args.out_dir, f"ocrvqa_raw_{split}.jsonl")
    kept = 0
    total = 0
    skipped_no_question = 0
    skipped_no_answer = 0
    skipped_no_image = 0
    records: List[Dict[str, Any]] = []
    first_keys_logged = False

    for sample in _iter_dataset(
        args.dataset, args.config, split, args.streaming, args.shuffle, args.seed
    ):
        total += 1
        question_value = _get_first(sample, ["question", "query", "text", "caption", "prompt"])
        if question_value is None:
            question_value = _find_by_terms(sample, ("question", "query", "prompt"))
        question = _ensure_text(question_value, "")
        answer = _extract_answer(sample, "")
        if not question or not answer:
            if not question:
                skipped_no_question += 1
            if not answer:
                skipped_no_answer += 1
            continue
        image_value = None
        if args.image_key:
            image_value = sample.get(args.image_key)
        if image_value is None:
            image_value = _get_first(
                sample,
                ["image", "img", "image_path", "image_file", "image_filename", "file_name", "filename"],
            )
        if image_value is None:
            image_value = _find_by_terms(sample, ("image", "img"))
        image = _load_image(
            image_value,
            dataset_name=args.dataset,
            download_images=args.download_images,
            repo_prefix=args.image_repo_prefix,
        )
        if image is None:
            skipped_no_image += 1
            if args.debug_keys and not first_keys_logged:
                logging.info("Sample keys: %s", sorted(list(sample.keys())))
                for key in sorted(sample.keys()):
                    key_l = str(key).lower()
                    if "image" in key_l or "img" in key_l or "question" in key_l or "answer" in key_l:
                        value = sample.get(key)
                        logging.info(
                            "Sample[%s]: type=%s value=%s",
                            key,
                            type(value).__name__,
                            str(value)[:200],
                        )
                first_keys_logged = True
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
    logging.info(
        "Saved %d samples to %s (split=%s). Skipped: no_question=%d no_answer=%d no_image=%d",
        kept,
        out_jsonl,
        split,
        skipped_no_question,
        skipped_no_answer,
        skipped_no_image,
    )


if __name__ == "__main__":
    main()
