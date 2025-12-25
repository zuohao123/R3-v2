"""Build unified JSONL files for ScreenQA, ChartQA, and InfoVQA."""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _ensure_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, list):
        value = value[0] if value else fallback
    if isinstance(value, dict):
        value = value.get("text", value)
    text = str(value).strip()
    return text if text else fallback


def _normalize_answer(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, list):
        value = value[0] if value else fallback
    if isinstance(value, dict):
        for key in ("answer", "answers", "label", "text"):
            if key in value and value[key]:
                value = value[key]
                break
    text = str(value).strip()
    return text if text else fallback


def _first_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _apply_prefix(image_path: str, prefix: Optional[str]) -> str:
    if not prefix:
        return image_path
    return os.path.join(prefix, image_path)


def build_screenqa_unified(raw_dir: str, out_dir: str, image_prefix: Optional[str] = None) -> None:
    """Convert ScreenQA raw JSONL files into unified JSONL format."""
    train_path = os.path.join(raw_dir, "screenqa_raw_train.jsonl")
    val_path = _first_existing(
        [
            os.path.join(raw_dir, "screenqa_raw_validation.jsonl"),
            os.path.join(raw_dir, "screenqa_raw_val.jsonl"),
            os.path.join(raw_dir, "screenqa_raw_test.jsonl"),
        ]
    )

    train_records = _read_jsonl(train_path)
    unified_train = []
    for raw in train_records:
        question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
        answer = _normalize_answer(raw.get("answer", ""), "[MISSING_ANSWER]")
        unified_train.append(
            {
                "image_path": _apply_prefix(raw["image_path"], image_prefix),
                "question": question,
                "answer": answer,
                "pseudo_text": f"SCREENQA_CONTEXT: {question} [ANSWER_HINT]",
            }
        )
    _write_jsonl(os.path.join(out_dir, "screenqa_unified_train.jsonl"), unified_train)
    logging.info("Wrote ScreenQA train: %d", len(unified_train))

    if val_path and os.path.exists(val_path):
        val_records = _read_jsonl(val_path)
        unified_val = []
        for raw in val_records:
            question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
            answer = _normalize_answer(raw.get("answer", ""), "[MISSING_ANSWER]")
            unified_val.append(
                {
                    "image_path": _apply_prefix(raw["image_path"], image_prefix),
                    "question": question,
                    "answer": answer,
                    "pseudo_text": f"SCREENQA_CONTEXT: {question} [ANSWER_HINT]",
                }
            )
        _write_jsonl(os.path.join(out_dir, "screenqa_unified_val.jsonl"), unified_val)
        logging.info("Wrote ScreenQA val: %d", len(unified_val))
    else:
        logging.warning("ScreenQA validation file not found; skipping val split.")


def build_chartqa_unified(raw_dir: str, out_dir: str, image_prefix: Optional[str] = None) -> None:
    """Convert ChartQA raw JSONL files into unified JSONL format."""
    train_path = os.path.join(raw_dir, "chartqa_raw_train.jsonl")
    eval_path = _first_existing(
        [
            os.path.join(raw_dir, "chartqa_raw_validation.jsonl"),
            os.path.join(raw_dir, "chartqa_raw_val.jsonl"),
            os.path.join(raw_dir, "chartqa_raw_test.jsonl"),
        ]
    )

    train_records = _read_jsonl(train_path)
    unified_train = []
    for raw in train_records:
        question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
        answer = _normalize_answer(raw.get("answer", ""), "[MISSING_ANSWER]")
        unified_train.append(
            {
                "image_path": _apply_prefix(raw["image_path"], image_prefix),
                "question": question,
                "answer": answer,
                "pseudo_text": f"CHARTQA_CONTEXT: {question}",
            }
        )
    _write_jsonl(os.path.join(out_dir, "chartqa_unified_train.jsonl"), unified_train)
    logging.info("Wrote ChartQA train: %d", len(unified_train))

    if eval_path and os.path.exists(eval_path):
        eval_records = _read_jsonl(eval_path)
        unified_eval = []
        for raw in eval_records:
            question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
            answer = _normalize_answer(raw.get("answer", ""), "[MISSING_ANSWER]")
            unified_eval.append(
                {
                    "image_path": _apply_prefix(raw["image_path"], image_prefix),
                    "question": question,
                    "answer": answer,
                    "pseudo_text": f"CHARTQA_CONTEXT: {question}",
                }
            )
        _write_jsonl(os.path.join(out_dir, "chartqa_unified_val.jsonl"), unified_eval)
        logging.info("Wrote ChartQA val/test: %d", len(unified_eval))
    else:
        logging.warning("ChartQA validation/test file not found; skipping val split.")


def build_infovqa_unified(raw_dir: str, out_dir: str, image_prefix: Optional[str] = None) -> None:
    """Convert InfoVQA raw JSONL files into unified JSONL format."""
    train_path = os.path.join(raw_dir, "infovqa_raw_train.jsonl")
    eval_path = _first_existing(
        [
            os.path.join(raw_dir, "infovqa_raw_val.jsonl"),
            os.path.join(raw_dir, "infovqa_raw_validation.jsonl"),
            os.path.join(raw_dir, "infovqa_raw_test.jsonl"),
        ]
    )

    train_records = _read_jsonl(train_path)
    unified_train = []
    for raw in train_records:
        question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
        answer = _normalize_answer(raw.get("answer", ""), "[MISSING_ANSWER]")
        unified_train.append(
            {
                "image_path": _apply_prefix(raw["image_path"], image_prefix),
                "question": question,
                "answer": answer,
                "pseudo_text": (
                    f"INFOVQA_CONTEXT: {question} [PREDICTED_TYPE:INFOGRAPHIC]"
                ),
            }
        )
    _write_jsonl(os.path.join(out_dir, "infovqa_unified_train.jsonl"), unified_train)
    logging.info("Wrote InfoVQA train: %d", len(unified_train))

    if eval_path and os.path.exists(eval_path):
        eval_records = _read_jsonl(eval_path)
        unified_eval = []
        for raw in eval_records:
            question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
            answer = _normalize_answer(raw.get("answer", ""), "[MISSING_ANSWER]")
            unified_eval.append(
                {
                    "image_path": _apply_prefix(raw["image_path"], image_prefix),
                    "question": question,
                    "answer": answer,
                    "pseudo_text": (
                        f"INFOVQA_CONTEXT: {question} [PREDICTED_TYPE:INFOGRAPHIC]"
                    ),
                }
            )
        out_name = (
            "infovqa_unified_test.jsonl"
            if eval_path.endswith("_test.jsonl")
            else "infovqa_unified_val.jsonl"
        )
        _write_jsonl(os.path.join(out_dir, out_name), unified_eval)
        logging.info("Wrote InfoVQA eval: %s (%d)", out_name, len(unified_eval))
    else:
        logging.warning("InfoVQA eval file not found; skipping val/test split.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified JSONL datasets.")
    parser.add_argument("--dataset", choices=["screenqa", "chartqa", "infovqa"], required=True)
    parser.add_argument("--raw_dir", required=True, help="Raw dataset directory")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--image_prefix",
        default=None,
        help="Optional prefix to prepend to image_path (useful for multi-dataset merges)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == "screenqa":
        build_screenqa_unified(args.raw_dir, args.out_dir, image_prefix=args.image_prefix)
    elif args.dataset == "chartqa":
        build_chartqa_unified(args.raw_dir, args.out_dir, image_prefix=args.image_prefix)
    else:
        build_infovqa_unified(args.raw_dir, args.out_dir, image_prefix=args.image_prefix)


if __name__ == "__main__":
    main()
