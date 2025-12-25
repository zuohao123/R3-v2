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


def _read_jsonl_shard(path: str, num_shards: int, shard_id: int) -> tuple[List[Dict[str, Any]], int]:
    if num_shards <= 1:
        records = _read_jsonl(path)
        return records, len(records)
    records: List[Dict[str, Any]] = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if (total % num_shards) == shard_id:
                records.append(json.loads(line))
            total += 1
    return records, total


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
    print(f"1value: {value}")
    if isinstance(value, dict):
        for key in ("answer", "answers", "label", "text", "full_answer", "ground_truth"):
            if key in value and value[key] not in (None, ""):
                value = value[key]
                break
    print(f"2value: {value}")
    if isinstance(value, list):
        value = value[0] if value else fallback
    print(f"3value: {value}")
    if isinstance(value, dict):
        for key in ("answer", "answers", "label", "text", "full_answer", "ground_truth"):
            if key in value and value[key] not in (None, ""):
                value = value[key]
                break
        if isinstance(value, list):
            value = value[0] if value else fallback
    print(f"4value: {value}")
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
            value = _normalize_answer(raw[key], fallback)
            if value != fallback:
                return value
    return fallback


def _summarize_value(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        if max_chars > 0 and len(value) > max_chars:
            return value[:max_chars] + "..."
        return value
    if isinstance(value, dict):
        summary: Dict[str, Any] = {}
        for key, val in value.items():
            summary[key] = _summarize_value(val, max_chars)
            if len(summary) >= 6:
                summary["..."] = "..."
                break
        return summary
    if isinstance(value, list):
        if not value:
            return []
        return [_summarize_value(value[0], max_chars)]
    return value


def _summarize_record(record: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in record.items():
        if key.lower() in {"image", "image_bytes"}:
            continue
        summary[key] = _summarize_value(value, max_chars)
        if len(summary) >= 10:
            summary["..."] = "..."
            break
    return summary


def _extract_answer_debug(
    raw: Dict[str, Any], fallback: str
) -> tuple[str, Optional[str], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
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
        if key not in raw or raw[key] in (None, ""):
            continue
        normalized = _normalize_answer(raw[key], fallback)
        candidates.append({"key": key, "normalized": normalized, "raw_type": type(raw[key]).__name__})
        if normalized != fallback:
            return normalized, key, candidates
    return fallback, None, candidates


def _first_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _apply_prefix(image_path: str, prefix: Optional[str]) -> str:
    if not prefix:
        return image_path
    return os.path.join(prefix, image_path)


def _shard_suffix(num_shards: int, shard_id: int) -> str:
    return f".shard{shard_id}" if num_shards > 1 else ""


def _merge_shards(out_dir: str, dataset: str, split: str, num_shards: int) -> bool:
    out_path = os.path.join(out_dir, f"{dataset}_unified_{split}.jsonl")
    shard_paths = [
        os.path.join(out_dir, f"{dataset}_unified_{split}.shard{i}.jsonl")
        for i in range(num_shards)
    ]
    existing = [path for path in shard_paths if os.path.exists(path)]
    if not existing:
        return False
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        for shard_path in shard_paths:
            if not os.path.exists(shard_path):
                logging.warning("Missing shard: %s", shard_path)
                continue
            with open(shard_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line.rstrip("\n") + "\n")
    logging.info("Merged %s into %s", split, out_path)
    return True


def _load_ocr_cache(path: Optional[str]) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if not path:
        return cache
    if not os.path.exists(path):
        logging.warning("OCR cache not found: %s", path)
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


def _lookup_ocr(
    image_path: str, image_prefix: Optional[str], cache: Dict[str, str], max_chars: int
) -> str:
    if not cache:
        return ""
    candidates = [image_path]
    if image_prefix:
        candidates.append(os.path.join(image_prefix, image_path))
    for key in candidates:
        text = cache.get(key)
        if text:
            return text[:max_chars] if max_chars > 0 else text
    return ""


def _compose_pseudo_text(tag: str, question: str, ocr_text: str, extra: Optional[str] = None) -> str:
    parts = [f"{tag}: {question}"]
    if extra:
        parts.append(extra)
    if ocr_text:
        parts.append(f"OCR: {ocr_text}")
    return " ".join(part for part in parts if part).strip()


def build_screenqa_unified(
    raw_dir: str,
    out_dir: str,
    image_prefix: Optional[str] = None,
    ocr_cache: Optional[str] = None,
    ocr_max_chars: int = 1200,
    num_shards: int = 1,
    shard_id: int = 0,
    log_every: int = 0,
    debug_samples: int = 0,
    debug_max_chars: int = 240,
) -> None:
    """Convert ScreenQA raw JSONL files into unified JSONL format."""
    train_path = os.path.join(raw_dir, "screenqa_raw_train.jsonl")
    val_path = _first_existing(
        [
            os.path.join(raw_dir, "screenqa_raw_validation.jsonl"),
            os.path.join(raw_dir, "screenqa_raw_val.jsonl"),
            os.path.join(raw_dir, "screenqa_raw_test.jsonl"),
        ]
    )

    ocr_map = _load_ocr_cache(ocr_cache)
    suffix = _shard_suffix(num_shards, shard_id)
    train_records, train_total = _read_jsonl_shard(train_path, num_shards, shard_id)
    train_records = train_records[:2]
    unified_train = []
    missing_train = 0
    for idx, raw in enumerate(train_records):
        question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
        print(f"raw:{raw}")
        answer = _extract_answer(raw, "")
        print(f"answer:{answer}")
        if debug_samples > 0 and idx < debug_samples:
            dbg_answer, dbg_key, dbg_candidates = _extract_answer_debug(raw, "[MISSING_ANSWER]")
            logging.info(
                "DEBUG ScreenQA train idx %d raw=%s",
                idx,
                _summarize_record(raw, debug_max_chars),
            )
            logging.info(
                "DEBUG ScreenQA train idx %d answer_candidates=%s selected_key=%s final=%s",
                idx,
                dbg_candidates,
                dbg_key,
                _summarize_value(dbg_answer, debug_max_chars),
            )
        if not answer:
            missing_train += 1
            continue
        ocr_text = _lookup_ocr(raw.get("image_path", ""), image_prefix, ocr_map, ocr_max_chars)
        unified_train.append(
            {
                "image_path": _apply_prefix(raw["image_path"], image_prefix),
                "question": question,
                "answer": answer,
                "ocr_text": ocr_text,
                    "pseudo_text": _compose_pseudo_text(
                        "SCREENQA_CONTEXT", question, ocr_text, extra="[ANSWER_HINT]"
                    ),
                }
            )
        if log_every and ((idx + 1) % log_every == 0 or (idx + 1) == len(train_records)):
            logging.info("ScreenQA train shard %d: %d/%d", shard_id, idx + 1, len(train_records))
    _write_jsonl(os.path.join(out_dir, f"screenqa_unified_train{suffix}.jsonl"), unified_train)
    logging.info(
        "Wrote ScreenQA train shard %d: %d (shard total %d, raw total %d)",
        shard_id,
        len(unified_train),
        len(train_records),
        train_total,
    )
    if missing_train:
        logging.warning(
            "ScreenQA train skipped %d/%d samples with missing answers.",
            missing_train,
            len(train_records),
        )

    if val_path and os.path.exists(val_path):
        val_records, val_total = _read_jsonl_shard(val_path, num_shards, shard_id)
        unified_val = []
        missing_val = 0
        for idx, raw in enumerate(val_records):
            question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
            answer = _extract_answer(raw, "[MISSING_ANSWER]")
            if debug_samples > 0 and idx < debug_samples:
                dbg_answer, dbg_key, dbg_candidates = _extract_answer_debug(raw, "[MISSING_ANSWER]")
                logging.info(
                    "DEBUG ScreenQA val idx %d raw=%s",
                    idx,
                    _summarize_record(raw, debug_max_chars),
                )
                logging.info(
                    "DEBUG ScreenQA val idx %d answer_candidates=%s selected_key=%s final=%s",
                    idx,
                    dbg_candidates,
                    dbg_key,
                    _summarize_value(dbg_answer, debug_max_chars),
                )
            if answer == "[MISSING_ANSWER]":
                missing_val += 1
            ocr_text = _lookup_ocr(raw.get("image_path", ""), image_prefix, ocr_map, ocr_max_chars)
            unified_val.append(
                {
                    "image_path": _apply_prefix(raw["image_path"], image_prefix),
                    "question": question,
                    "answer": answer,
                    "ocr_text": ocr_text,
                    "pseudo_text": _compose_pseudo_text(
                        "SCREENQA_CONTEXT", question, ocr_text, extra="[ANSWER_HINT]"
                    ),
                }
            )
            if log_every and ((idx + 1) % log_every == 0 or (idx + 1) == len(val_records)):
                logging.info("ScreenQA val shard %d: %d/%d", shard_id, idx + 1, len(val_records))
        _write_jsonl(os.path.join(out_dir, f"screenqa_unified_val{suffix}.jsonl"), unified_val)
        logging.info(
            "Wrote ScreenQA val shard %d: %d (shard total %d, raw total %d)",
            shard_id,
            len(unified_val),
            len(val_records),
            val_total,
        )
        if missing_val:
            logging.warning(
                "ScreenQA val has %d/%d samples with missing answers.",
                missing_val,
                len(val_records),
            )
    else:
        logging.warning("ScreenQA validation file not found; skipping val split.")


def build_chartqa_unified(
    raw_dir: str,
    out_dir: str,
    image_prefix: Optional[str] = None,
    ocr_cache: Optional[str] = None,
    ocr_max_chars: int = 1200,
    num_shards: int = 1,
    shard_id: int = 0,
    log_every: int = 0,
    debug_samples: int = 0,
    debug_max_chars: int = 240,
) -> None:
    """Convert ChartQA raw JSONL files into unified JSONL format."""
    train_path = os.path.join(raw_dir, "chartqa_raw_train.jsonl")
    eval_path = _first_existing(
        [
            os.path.join(raw_dir, "chartqa_raw_validation.jsonl"),
            os.path.join(raw_dir, "chartqa_raw_val.jsonl"),
            os.path.join(raw_dir, "chartqa_raw_test.jsonl"),
        ]
    )

    ocr_map = _load_ocr_cache(ocr_cache)
    suffix = _shard_suffix(num_shards, shard_id)
    train_records, train_total = _read_jsonl_shard(train_path, num_shards, shard_id)
    unified_train = []
    missing_train = 0
    for idx, raw in enumerate(train_records):
        question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
        answer = _extract_answer(raw, "")
        if debug_samples > 0 and idx < debug_samples:
            dbg_answer, dbg_key, dbg_candidates = _extract_answer_debug(raw, "[MISSING_ANSWER]")
            logging.info(
                "DEBUG ChartQA train idx %d raw=%s",
                idx,
                _summarize_record(raw, debug_max_chars),
            )
            logging.info(
                "DEBUG ChartQA train idx %d answer_candidates=%s selected_key=%s final=%s",
                idx,
                dbg_candidates,
                dbg_key,
                _summarize_value(dbg_answer, debug_max_chars),
            )
        if not answer:
            missing_train += 1
            continue
        ocr_text = _lookup_ocr(raw.get("image_path", ""), image_prefix, ocr_map, ocr_max_chars)
        unified_train.append(
            {
                "image_path": _apply_prefix(raw["image_path"], image_prefix),
                "question": question,
                "answer": answer,
                "ocr_text": ocr_text,
                "pseudo_text": _compose_pseudo_text("CHARTQA_CONTEXT", question, ocr_text),
            }
        )
        if log_every and ((idx + 1) % log_every == 0 or (idx + 1) == len(train_records)):
            logging.info("ChartQA train shard %d: %d/%d", shard_id, idx + 1, len(train_records))
    _write_jsonl(os.path.join(out_dir, f"chartqa_unified_train{suffix}.jsonl"), unified_train)
    logging.info(
        "Wrote ChartQA train shard %d: %d (shard total %d, raw total %d)",
        shard_id,
        len(unified_train),
        len(train_records),
        train_total,
    )
    if missing_train:
        logging.warning(
            "ChartQA train skipped %d/%d samples with missing answers.",
            missing_train,
            len(train_records),
        )

    if eval_path and os.path.exists(eval_path):
        eval_records, eval_total = _read_jsonl_shard(eval_path, num_shards, shard_id)
        unified_eval = []
        missing_eval = 0
        for idx, raw in enumerate(eval_records):
            question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
            answer = _extract_answer(raw, "[MISSING_ANSWER]")
            if debug_samples > 0 and idx < debug_samples:
                dbg_answer, dbg_key, dbg_candidates = _extract_answer_debug(raw, "[MISSING_ANSWER]")
                logging.info(
                    "DEBUG ChartQA val/test idx %d raw=%s",
                    idx,
                    _summarize_record(raw, debug_max_chars),
                )
                logging.info(
                    "DEBUG ChartQA val/test idx %d answer_candidates=%s selected_key=%s final=%s",
                    idx,
                    dbg_candidates,
                    dbg_key,
                    _summarize_value(dbg_answer, debug_max_chars),
                )
            if answer == "[MISSING_ANSWER]":
                missing_eval += 1
            ocr_text = _lookup_ocr(raw.get("image_path", ""), image_prefix, ocr_map, ocr_max_chars)
            unified_eval.append(
                {
                    "image_path": _apply_prefix(raw["image_path"], image_prefix),
                    "question": question,
                    "answer": answer,
                    "ocr_text": ocr_text,
                    "pseudo_text": _compose_pseudo_text(
                        "CHARTQA_CONTEXT", question, ocr_text
                    ),
                }
            )
            if log_every and ((idx + 1) % log_every == 0 or (idx + 1) == len(eval_records)):
                logging.info("ChartQA val/test shard %d: %d/%d", shard_id, idx + 1, len(eval_records))
        _write_jsonl(os.path.join(out_dir, f"chartqa_unified_val{suffix}.jsonl"), unified_eval)
        logging.info(
            "Wrote ChartQA val/test shard %d: %d (shard total %d, raw total %d)",
            shard_id,
            len(unified_eval),
            len(eval_records),
            eval_total,
        )
        if missing_eval:
            logging.warning(
                "ChartQA val/test has %d/%d samples with missing answers.",
                missing_eval,
                len(eval_records),
            )
    else:
        logging.warning("ChartQA validation/test file not found; skipping val split.")


def build_infovqa_unified(
    raw_dir: str,
    out_dir: str,
    image_prefix: Optional[str] = None,
    ocr_cache: Optional[str] = None,
    ocr_max_chars: int = 1200,
    num_shards: int = 1,
    shard_id: int = 0,
    log_every: int = 0,
    debug_samples: int = 0,
    debug_max_chars: int = 240,
) -> None:
    """Convert InfoVQA raw JSONL files into unified JSONL format."""
    train_path = os.path.join(raw_dir, "infovqa_raw_train.jsonl")
    eval_path = _first_existing(
        [
            os.path.join(raw_dir, "infovqa_raw_val.jsonl"),
            os.path.join(raw_dir, "infovqa_raw_validation.jsonl"),
            os.path.join(raw_dir, "infovqa_raw_test.jsonl"),
        ]
    )

    ocr_map = _load_ocr_cache(ocr_cache)
    suffix = _shard_suffix(num_shards, shard_id)
    train_records, train_total = _read_jsonl_shard(train_path, num_shards, shard_id)
    unified_train = []
    missing_train = 0
    for idx, raw in enumerate(train_records):
        question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
        answer = _extract_answer(raw, "")
        if debug_samples > 0 and idx < debug_samples:
            dbg_answer, dbg_key, dbg_candidates = _extract_answer_debug(raw, "[MISSING_ANSWER]")
            logging.info(
                "DEBUG InfoVQA train idx %d raw=%s",
                idx,
                _summarize_record(raw, debug_max_chars),
            )
            logging.info(
                "DEBUG InfoVQA train idx %d answer_candidates=%s selected_key=%s final=%s",
                idx,
                dbg_candidates,
                dbg_key,
                _summarize_value(dbg_answer, debug_max_chars),
            )
        if not answer:
            missing_train += 1
            continue
        ocr_text = _lookup_ocr(raw.get("image_path", ""), image_prefix, ocr_map, ocr_max_chars)
        unified_train.append(
            {
                "image_path": _apply_prefix(raw["image_path"], image_prefix),
                "question": question,
                "answer": answer,
                "ocr_text": ocr_text,
                "pseudo_text": _compose_pseudo_text(
                    "INFOVQA_CONTEXT",
                    question,
                    ocr_text,
                    extra="[PREDICTED_TYPE:INFOGRAPHIC]",
                ),
            }
        )
        if log_every and ((idx + 1) % log_every == 0 or (idx + 1) == len(train_records)):
            logging.info("InfoVQA train shard %d: %d/%d", shard_id, idx + 1, len(train_records))
    _write_jsonl(os.path.join(out_dir, f"infovqa_unified_train{suffix}.jsonl"), unified_train)
    logging.info(
        "Wrote InfoVQA train shard %d: %d (shard total %d, raw total %d)",
        shard_id,
        len(unified_train),
        len(train_records),
        train_total,
    )
    if missing_train:
        logging.warning(
            "InfoVQA train skipped %d/%d samples with missing answers.",
            missing_train,
            len(train_records),
        )

    if eval_path and os.path.exists(eval_path):
        eval_records, eval_total = _read_jsonl_shard(eval_path, num_shards, shard_id)
        unified_eval = []
        missing_eval = 0
        for idx, raw in enumerate(eval_records):
            question = _ensure_text(raw.get("question", ""), "[MISSING_QUESTION]")
            answer = _extract_answer(raw, "[MISSING_ANSWER]")
            if debug_samples > 0 and idx < debug_samples:
                dbg_answer, dbg_key, dbg_candidates = _extract_answer_debug(raw, "[MISSING_ANSWER]")
                logging.info(
                    "DEBUG InfoVQA eval idx %d raw=%s",
                    idx,
                    _summarize_record(raw, debug_max_chars),
                )
                logging.info(
                    "DEBUG InfoVQA eval idx %d answer_candidates=%s selected_key=%s final=%s",
                    idx,
                    dbg_candidates,
                    dbg_key,
                    _summarize_value(dbg_answer, debug_max_chars),
                )
            if answer == "[MISSING_ANSWER]":
                missing_eval += 1
            ocr_text = _lookup_ocr(raw.get("image_path", ""), image_prefix, ocr_map, ocr_max_chars)
            unified_eval.append(
                {
                    "image_path": _apply_prefix(raw["image_path"], image_prefix),
                    "question": question,
                    "answer": answer,
                    "ocr_text": ocr_text,
                    "pseudo_text": _compose_pseudo_text(
                        "INFOVQA_CONTEXT",
                        question,
                        ocr_text,
                        extra="[PREDICTED_TYPE:INFOGRAPHIC]",
                    ),
                }
            )
            if log_every and ((idx + 1) % log_every == 0 or (idx + 1) == len(eval_records)):
                logging.info("InfoVQA eval shard %d: %d/%d", shard_id, idx + 1, len(eval_records))
        out_name = (
            "infovqa_unified_test.jsonl"
            if eval_path.endswith("_test.jsonl")
            else "infovqa_unified_val.jsonl"
        )
        out_name = out_name.replace(".jsonl", f"{suffix}.jsonl")
        _write_jsonl(os.path.join(out_dir, out_name), unified_eval)
        logging.info(
            "Wrote InfoVQA eval shard %d: %s (%d, shard total %d, raw total %d)",
            shard_id,
            out_name,
            len(unified_eval),
            len(eval_records),
            eval_total,
        )
        if missing_eval:
            logging.warning(
                "InfoVQA eval has %d/%d samples with missing answers.",
                missing_eval,
                len(eval_records),
            )
    else:
        logging.warning("InfoVQA eval file not found; skipping val/test split.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified JSONL datasets.")
    parser.add_argument("--dataset", choices=["screenqa", "chartqa", "infovqa"], required=True)
    parser.add_argument("--raw_dir", default=None, help="Raw dataset directory")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--image_prefix",
        default=None,
        help="Optional prefix to prepend to image_path (useful for multi-dataset merges)",
    )
    parser.add_argument("--ocr_cache", default=None, help="Optional OCR cache JSONL path")
    parser.add_argument("--ocr_max_chars", type=int, default=1200)
    parser.add_argument("--debug_samples", type=int, default=0)
    parser.add_argument("--debug_max_chars", type=int, default=240)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--merge_shards", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge_shards:
        splits = {
            "screenqa": ["train", "val"],
            "chartqa": ["train", "val"],
            "infovqa": ["train", "val", "test"],
        }[args.dataset]
        merged_any = False
        for split in splits:
            merged_any = _merge_shards(args.out_dir, args.dataset, split, args.merge_shards) or merged_any
        if not merged_any:
            logging.warning("No shards found to merge for %s.", args.dataset)
        return

    if args.raw_dir is None:
        parser.error("--raw_dir is required unless --merge_shards is set.")
    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1.")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        parser.error("--shard_id must be in [0, num_shards).")

    if args.dataset == "screenqa":
        build_screenqa_unified(
            args.raw_dir,
            args.out_dir,
            image_prefix=args.image_prefix,
            ocr_cache=args.ocr_cache,
            ocr_max_chars=args.ocr_max_chars,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            log_every=args.log_every,
            debug_samples=args.debug_samples,
            debug_max_chars=args.debug_max_chars,
        )
    elif args.dataset == "chartqa":
        build_chartqa_unified(
            args.raw_dir,
            args.out_dir,
            image_prefix=args.image_prefix,
            ocr_cache=args.ocr_cache,
            ocr_max_chars=args.ocr_max_chars,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            log_every=args.log_every,
            debug_samples=args.debug_samples,
            debug_max_chars=args.debug_max_chars,
        )
    else:
        build_infovqa_unified(
            args.raw_dir,
            args.out_dir,
            image_prefix=args.image_prefix,
            ocr_cache=args.ocr_cache,
            ocr_max_chars=args.ocr_max_chars,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            log_every=args.log_every,
            debug_samples=args.debug_samples,
            debug_max_chars=args.debug_max_chars,
        )


if __name__ == "__main__":
    main()
