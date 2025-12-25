"""Rebuild unified JSONL files and retrieval indices from existing raw JSONL."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterable, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.preprocess.build_unified_json import (
    build_chartqa_unified,
    build_infovqa_unified,
    build_screenqa_unified,
)


def _first_existing(paths: Iterable[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _concat_jsonl(paths: List[str], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for path in paths:
            if not os.path.exists(path):
                logging.warning("Missing JSONL: %s", path)
                continue
            with open(path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line if line.endswith("\n") else line + "\n")
                        total += 1
    return total


def _run_build_indices(
    jsonl: str,
    image_root: str,
    out_dir: str,
    image_encoder: str,
    text_encoder: str,
    batch_size: int,
    num_shards: int,
    shard_id: Optional[int],
    merge_shards: Optional[int],
    device: Optional[str],
    log_every: int,
) -> None:
    import scripts.build_indices as build_indices

    argv = ["build_indices.py", "--out_dir", out_dir]
    if merge_shards:
        argv += ["--merge_shards", str(merge_shards)]
    else:
        argv += [
            "--jsonl",
            jsonl,
            "--image_root",
            image_root,
            "--image_encoder",
            image_encoder,
            "--text_encoder",
            text_encoder,
            "--batch_size",
            str(batch_size),
            "--num_shards",
            str(num_shards),
            "--log_every",
            str(log_every),
        ]
        if shard_id is not None:
            argv += ["--shard_id", str(shard_id)]
        if device:
            argv += ["--device", device]

    old_argv = sys.argv
    sys.argv = argv
    try:
        build_indices.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild unified JSONL and indices from existing raw JSONL."
    )
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--out_dir", default="data/unified")
    parser.add_argument("--image_root", default="data/raw")
    parser.add_argument("--index_dir", default="indices")
    parser.add_argument("--skip_indices", action="store_true")
    parser.add_argument("--ocr_root", default=None, help="Directory with OCR caches")
    parser.add_argument("--screenqa_ocr", default=None)
    parser.add_argument("--chartqa_ocr", default=None)
    parser.add_argument("--infovqa_ocr", default=None)
    parser.add_argument("--ocr_max_chars", type=int, default=1200)
    parser.add_argument("--image_encoder", default="models/clip-vit-b32-laion2B")
    parser.add_argument("--text_encoder", default="models/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--merge_shards", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log_every", type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    screenqa_raw = os.path.join(args.raw_root, "screenqa")
    chartqa_raw = os.path.join(args.raw_root, "chartqa")
    infovqa_raw = os.path.join(args.raw_root, "infovqa")

    screenqa_ocr = args.screenqa_ocr
    chartqa_ocr = args.chartqa_ocr
    infovqa_ocr = args.infovqa_ocr
    if args.ocr_root:
        screenqa_ocr = screenqa_ocr or os.path.join(args.ocr_root, "screenqa_ocr.jsonl")
        chartqa_ocr = chartqa_ocr or os.path.join(args.ocr_root, "chartqa_ocr.jsonl")
        infovqa_ocr = infovqa_ocr or os.path.join(args.ocr_root, "infovqa_ocr.jsonl")

    if os.path.exists(screenqa_raw):
        build_screenqa_unified(
            screenqa_raw,
            args.out_dir,
            image_prefix="screenqa",
            ocr_cache=screenqa_ocr,
            ocr_max_chars=args.ocr_max_chars,
        )
    else:
        logging.warning("ScreenQA raw dir not found: %s", screenqa_raw)

    if os.path.exists(chartqa_raw):
        build_chartqa_unified(
            chartqa_raw,
            args.out_dir,
            image_prefix="chartqa",
            ocr_cache=chartqa_ocr,
            ocr_max_chars=args.ocr_max_chars,
        )
    else:
        logging.warning("ChartQA raw dir not found: %s", chartqa_raw)

    if os.path.exists(infovqa_raw):
        build_infovqa_unified(
            infovqa_raw,
            args.out_dir,
            image_prefix="infovqa",
            ocr_cache=infovqa_ocr,
            ocr_max_chars=args.ocr_max_chars,
        )
    else:
        logging.warning("InfoVQA raw dir not found: %s", infovqa_raw)

    train_paths = [
        os.path.join(args.out_dir, "screenqa_unified_train.jsonl"),
        os.path.join(args.out_dir, "chartqa_unified_train.jsonl"),
        os.path.join(args.out_dir, "infovqa_unified_train.jsonl"),
    ]
    val_paths = [
        os.path.join(args.out_dir, "screenqa_unified_val.jsonl"),
        os.path.join(args.out_dir, "chartqa_unified_val.jsonl"),
        _first_existing(
            [
                os.path.join(args.out_dir, "infovqa_unified_test.jsonl"),
                os.path.join(args.out_dir, "infovqa_unified_val.jsonl"),
            ]
        ),
    ]
    val_paths = [p for p in val_paths if p]

    train_out = os.path.join(args.out_dir, "train.jsonl")
    val_out = os.path.join(args.out_dir, "val.jsonl")
    train_count = _concat_jsonl(train_paths, train_out)
    val_count = _concat_jsonl(val_paths, val_out)
    logging.info("Merged train.jsonl (%d) and val.jsonl (%d).", train_count, val_count)

    if args.skip_indices:
        return

    if args.num_shards > 1 and args.merge_shards is None:
        if args.shard_id is None and os.environ.get("LOCAL_RANK") is None:
            logging.error(
                "--num_shards>1 requires --shard_id or running under torchrun."
            )
            return

    _run_build_indices(
        train_out,
        image_root=args.image_root,
        out_dir=args.index_dir,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        batch_size=args.batch_size,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        merge_shards=args.merge_shards,
        device=args.device,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
