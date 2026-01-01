"""Build WebSRC unified JSONL with pseudo_text and OCR."""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional


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


def _load_ocr_cache(path: Optional[str]) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record.get("image_path", "")).strip()
            if not key:
                continue
            text = record.get("ocr_text") or record.get("text") or record.get("value") or ""
            text = str(text).strip()
            if text:
                cache[key] = text
    return cache


def _compose_pseudo_text(tag: str, question: str, ocr_text: str) -> str:
    parts = [f"{tag}: {question}"]
    if ocr_text:
        parts.append(f"OCR: {ocr_text}")
    return " ".join(parts).strip()


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WebSRC unified JSONL.")
    parser.add_argument("--raw_jsonl", required=True)
    parser.add_argument("--ocr_cache", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--ocr_max_chars", type=int, default=1200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ocr_map = _load_ocr_cache(args.ocr_cache)
    records: List[Dict[str, Any]] = []
    kept = 0

    for raw in _iter_jsonl(args.raw_jsonl):
        question = _ensure_text(raw.get("question"), "[MISSING_QUESTION]")
        answer = _extract_answer(raw, "")
        if not answer:
            continue
        image_path = str(raw.get("image_path", "")).strip()
        if not image_path:
            continue
        ocr_text = ocr_map.get(image_path, "")
        if args.ocr_max_chars > 0:
            ocr_text = ocr_text[: args.ocr_max_chars]
        pseudo_text = _compose_pseudo_text("WEBSRC_CONTEXT", question, ocr_text)
        records.append(
            {
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "ocr_text": ocr_text,
                "pseudo_text": pseudo_text,
            }
        )
        kept += 1
        if args.max_samples and kept >= args.max_samples:
            break

    _write_jsonl(args.out_jsonl, records)
    logging.info("Saved %d samples to %s", kept, args.out_jsonl)


if __name__ == "__main__":
    main()
