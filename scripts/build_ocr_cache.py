"""Build OCR cache JSONL from raw dataset JSONL files."""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SPACE_RE = re.compile(r"\s+")
_PUNCT_FIX_RE = re.compile(r"\s+([,.;:!?])")
_TOKEN_CLEAN_RE = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")


@dataclass
class OCRItem:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)


def _iter_jsonl(paths: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def _resolve_image_path(path: str, image_root: str) -> str:
    if image_root and not os.path.isabs(path):
        return os.path.join(image_root, path)
    return path


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = _SPACE_RE.sub(" ", text)
    text = _PUNCT_FIX_RE.sub(r"\1", text)
    return text


def _valid_token(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if _CJK_RE.search(text):
        return True
    cleaned = _TOKEN_CLEAN_RE.sub("", text)
    if len(cleaned) < 2:
        return False
    return any(ch.isalnum() for ch in cleaned)


def _line_group(items: List[OCRItem]) -> List[List[OCRItem]]:
    if not items:
        return []
    items = sorted(items, key=lambda x: (x.cy, x.cx))
    heights = [item.height for item in items]
    line_tol = max(10.0, median(heights) * 0.6)
    lines: List[List[OCRItem]] = []
    for item in items:
        if not lines or abs(item.cy - lines[-1][0].cy) > line_tol:
            lines.append([item])
        else:
            lines[-1].append(item)
    for line in lines:
        line.sort(key=lambda x: x.cx)
    return lines


def _dedup_lines(lines: List[str]) -> List[str]:
    seen = set()
    out = []
    for line in lines:
        key = _TOKEN_CLEAN_RE.sub("", line.lower())
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _postprocess(items: List[OCRItem], max_chars: int) -> str:
    if not items:
        return ""
    lines = []
    for line_items in _line_group(items):
        tokens = [_normalize_text(item.text) for item in line_items if _valid_token(item.text)]
        if not tokens:
            continue
        line = " ".join(tokens)
        line = _normalize_text(line)
        if line:
            lines.append(line)
    lines = _dedup_lines(lines)
    text = " ".join(lines)
    text = _normalize_text(text)
    if max_chars > 0:
        text = text[:max_chars]
    return text


class OCREngine:
    def __init__(self, engine: str, lang: str, use_gpu: bool, use_angle_cls: bool) -> None:
        self.engine = engine
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        if engine == "paddleocr":
            from paddleocr import PaddleOCR

            self.backend = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False,
            )
        elif engine == "easyocr":
            import easyocr

            self.backend = easyocr.Reader([lang], gpu=use_gpu)
        elif engine == "tesseract":
            import pytesseract

            self.backend = pytesseract
        else:
            raise ValueError(f"Unknown OCR engine: {engine}")

    def extract(self, image: Image.Image) -> List[OCRItem]:
        if self.engine == "paddleocr":
            result = self.backend.ocr(np.array(image), cls=self.use_angle_cls)
            return _parse_paddle(result)
        if self.engine == "easyocr":
            result = self.backend.readtext(np.array(image))
            return _parse_easyocr(result)
        return _parse_tesseract(self.backend, image)


def _parse_paddle(result: Any) -> List[OCRItem]:
    items: List[OCRItem] = []
    if not result:
        return items
    for line in result[0]:
        box, (text, conf) = line
        x0 = min(pt[0] for pt in box)
        y0 = min(pt[1] for pt in box)
        x1 = max(pt[0] for pt in box)
        y1 = max(pt[1] for pt in box)
        items.append(OCRItem(text=text, conf=float(conf), x0=x0, y0=y0, x1=x1, y1=y1))
    return items


def _parse_easyocr(result: Any) -> List[OCRItem]:
    items: List[OCRItem] = []
    for box, text, conf in result:
        x0 = min(pt[0] for pt in box)
        y0 = min(pt[1] for pt in box)
        x1 = max(pt[0] for pt in box)
        y1 = max(pt[1] for pt in box)
        items.append(OCRItem(text=text, conf=float(conf), x0=x0, y0=y0, x1=x1, y1=y1))
    return items


def _parse_tesseract(pytesseract, image: Image.Image) -> List[OCRItem]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    items: List[OCRItem] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = data["text"][i]
        try:
            conf = float(data["conf"][i]) / 100.0
        except Exception:
            conf = 0.0
        if not text:
            continue
        x0 = float(data["left"][i])
        y0 = float(data["top"][i])
        x1 = x0 + float(data["width"][i])
        y1 = y0 + float(data["height"][i])
        items.append(OCRItem(text=text, conf=conf, x0=x0, y0=y0, x1=x1, y1=y1))
    return items


def _write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OCR cache JSONL")
    parser.add_argument("--raw_dir", default=None, help="Raw dataset dir with *_raw_*.jsonl")
    parser.add_argument("--jsonl", nargs="+", default=None, help="Explicit JSONL paths")
    parser.add_argument("--image_root", default="", help="Image root for relative paths")
    parser.add_argument("--out_path", required=True, help="Output OCR cache JSONL")
    parser.add_argument("--engine", choices=["paddleocr", "easyocr", "tesseract"], default="paddleocr")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--use_angle_cls", action="store_true")
    parser.add_argument("--min_conf", type=float, default=0.5)
    parser.add_argument("--max_chars", type=int, default=1200)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    jsonl_paths: List[str] = []
    if args.raw_dir:
        jsonl_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*_raw_*.jsonl")))
    if args.jsonl:
        jsonl_paths.extend(args.jsonl)
    jsonl_paths = [p for p in jsonl_paths if os.path.exists(p)]
    if not jsonl_paths:
        raise ValueError("No JSONL files found. Provide --raw_dir or --jsonl.")

    engine = OCREngine(
        engine=args.engine,
        lang=args.lang,
        use_gpu=args.use_gpu,
        use_angle_cls=args.use_angle_cls,
    )

    seen = set()
    records: List[Dict[str, Any]] = []
    total = 0
    for idx, record in enumerate(_iter_jsonl(jsonl_paths)):
        if args.max_samples is not None and total >= args.max_samples:
            break
        image_path = record.get("image_path")
        if not image_path or image_path in seen:
            continue
        seen.add(image_path)
        img_path = _resolve_image_path(image_path, args.image_root)
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            continue
        items = engine.extract(image)
        items = [item for item in items if item.conf >= args.min_conf]
        ocr_text = _postprocess(items, args.max_chars)
        records.append({"image_path": image_path, "ocr_text": ocr_text})
        total += 1
        if args.log_every > 0 and total % args.log_every == 0:
            logging.info("OCR progress: %d", total)

    _write_jsonl(records, args.out_path)
    logging.info("Saved OCR cache: %s (%d entries)", args.out_path, len(records))


if __name__ == "__main__":
    main()
