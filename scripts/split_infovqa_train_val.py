"""Split InfoVQA train into train/val and optionally delete test."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split InfoVQA train into train/val.")
    parser.add_argument("--raw_dir", required=True, help="Raw InfoVQA directory")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep_test",
        action="store_true",
        help="Keep infovqa_raw_test.jsonl if present",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train_path = os.path.join(args.raw_dir, "infovqa_raw_train.jsonl")
    test_path = os.path.join(args.raw_dir, "infovqa_raw_test.jsonl")
    val_path = os.path.join(args.raw_dir, "infovqa_raw_val.jsonl")

    records = _read_jsonl(train_path)
    if not records:
        raise RuntimeError(f"No records found in {train_path}")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    val_count = int(len(records) * args.val_ratio)
    if args.val_ratio > 0 and val_count == 0:
        val_count = 1
    val_records = records[:val_count]
    train_records = records[val_count:]

    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)

    logging.info("Wrote train: %s (%d)", train_path, len(train_records))
    logging.info("Wrote val: %s (%d)", val_path, len(val_records))

    if not args.keep_test and os.path.exists(test_path):
        os.remove(test_path)
        logging.info("Deleted test file: %s", test_path)


if __name__ == "__main__":
    main()
