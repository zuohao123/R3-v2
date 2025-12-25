"""Download and export ScreenQA data from HuggingFace datasets."""
from __future__ import annotations

import argparse
import json
import logging
import os
from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset
from PIL import Image


def save_image(image: Image.Image, path: str) -> None:
    """Save a PIL image to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.convert("RGB").save(path)


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    """Write records to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _get_first(sample: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in sample and sample[key] not in (None, ""):
            return sample[key]
    return None


def _to_pil(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value
    if isinstance(image_value, dict):
        if "bytes" in image_value and image_value["bytes"] is not None:
            return Image.open(BytesIO(image_value["bytes"]))
        if "path" in image_value and image_value["path"]:
            return Image.open(image_value["path"])
    raise ValueError("Unsupported image format in dataset sample.")


def _normalize_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, list):
        value = value[0] if value else fallback
    if isinstance(value, dict):
        value = value.get("text", value)
    text = str(value).strip()
    return text if text else fallback


def _normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("answer", "answers", "label", "text", "full_answer", "ground_truth"):
            if key in value and value[key] not in (None, ""):
                value = value[key]
                break
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        for key in ("answer", "answers", "label", "text", "full_answer", "ground_truth"):
            if key in value and value[key] not in (None, ""):
                value = value[key]
                break
        if isinstance(value, list):
            value = value[0] if value else ""
    return str(value).strip()


def _process_sample(
    sample: Dict[str, Any],
    idx: int,
    split: str,
    images_dir: str,
) -> Dict[str, Any]:
    image_value = _get_first(sample, ["image", "img", "image_path", "image_file"])
    if image_value is None:
        return {"skip": True}
    image = _to_pil(image_value)
    question = _normalize_text(
        _get_first(sample, ["question", "query", "text"]), "[MISSING_QUESTION]"
    )
    answer = _normalize_answer(
        _get_first(
            sample,
            ["answer", "answers", "label", "gt_answer", "full_answer", "ground_truth"],
        )
    )
    image_name = f"{split}_{idx}.png"
    rel_path = os.path.join("images", image_name)
    save_image(image, os.path.join(images_dir, image_name))
    return {
        "image_path": rel_path,
        "question": question,
        "answer": answer,
        "split": split,
        "skip": False,
    }


def _not_skip(example: Dict[str, Any]) -> bool:
    return not example.get("skip", False)


def export_split(ds, split: str, out_dir: str, log_every: int, num_proc: int) -> None:
    """Export one split into images and JSONL."""
    images_dir = os.path.join(out_dir, "images")
    raw_path = os.path.join(out_dir, f"screenqa_raw_{split}.jsonl")
    total = len(ds)
    logging.info("Exporting %s split with %d samples", split, total)
    os.makedirs(images_dir, exist_ok=True)
    if num_proc > 1:
        processed = ds.map(
            _process_sample,
            with_indices=True,
            fn_kwargs={"split": split, "images_dir": images_dir},
            remove_columns=ds.column_names,
            num_proc=num_proc,
            load_from_cache_file=False,
            desc=f"Export {split}",
        )
        processed = processed.filter(_not_skip, num_proc=num_proc)
        processed = processed.remove_columns(["skip"])
        write_jsonl(raw_path, processed)
        logging.info("Wrote %s with %d samples", raw_path, len(processed))
        return

    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(ds):
        record = _process_sample(sample, idx, split, images_dir)
        if record.get("skip"):
            logging.warning("Missing image for %s sample %d, skipping.", split, idx)
            continue
        record.pop("skip", None)
        records.append(record)
        if log_every > 0 and ((idx + 1) % log_every == 0 or (idx + 1) == total):
            logging.info("Progress %s: %d/%d", split, idx + 1, total)
    write_jsonl(raw_path, records)
    logging.info("Wrote %s with %d samples", raw_path, len(records))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ScreenQA dataset.")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--log_every",
        type=int,
        default=500,
        help="Log progress every N samples",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of worker processes for preprocessing",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("rootsautomation/RICO-ScreenQA-Short")
    for split in ds.keys():
        logging.info("Processing split: %s", split)
        export_split(ds[split], split, args.out_dir, args.log_every, args.num_proc)


if __name__ == "__main__":
    main()
