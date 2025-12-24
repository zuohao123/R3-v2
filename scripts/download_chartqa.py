"""Download and export ChartQA data from HuggingFace datasets."""
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
    if isinstance(value, list):
        value = value[0] if value else ""
    if isinstance(value, dict):
        for key in ("answer", "answers", "label", "text"):
            if key in value and value[key]:
                value = value[key]
                break
    return str(value).strip()


def export_split(ds, split: str, out_dir: str) -> None:
    """Export one split into images and JSONL."""
    images_dir = os.path.join(out_dir, "images")
    raw_path = os.path.join(out_dir, f"chartqa_raw_{split}.jsonl")
    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(ds):
        image_value = _get_first(sample, ["image", "img", "image_path", "image_file"])
        if image_value is None:
            logging.warning("Missing image for %s sample %d, skipping.", split, idx)
            continue
        image = _to_pil(image_value)
        question = _normalize_text(
            _get_first(sample, ["question", "query", "text"]), "[MISSING_QUESTION]"
        )
        answer = _normalize_answer(
            _get_first(sample, ["answer", "answers", "label", "gt_answer"])
        )
        image_name = f"{split}_{idx}.png"
        rel_path = os.path.join("images", image_name)
        save_image(image, os.path.join(images_dir, image_name))
        records.append(
            {
                "image_path": rel_path,
                "question": question,
                "answer": answer,
                "split": split,
            }
        )
    write_jsonl(raw_path, records)
    logging.info("Wrote %s with %d samples", raw_path, len(records))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ChartQA dataset.")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("HuggingFaceM4/ChartQA")
    for split in ds.keys():
        logging.info("Processing split: %s", split)
        export_split(ds[split], split, args.out_dir)


if __name__ == "__main__":
    main()
