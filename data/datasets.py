"""Unified dataset and collator for multimodal QA."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class UnifiedQADatum:
    """Single QA record for multimodal QA."""
    image_path: str
    question: str
    answer: str
    pseudo_text: str


class UnifiedQADataset(Dataset):
    """Dataset reading unified JSONL files."""

    def __init__(
        self, jsonl_path: str, image_root: str = "", max_samples: Optional[int] = None
    ) -> None:
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.samples: List[UnifiedQADatum] = []
        self._load(max_samples=max_samples)

    def _load(self, max_samples: Optional[int]) -> None:
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self.samples.append(
                    UnifiedQADatum(
                        image_path=data["image_path"],
                        question=data["question"],
                        answer=data["answer"],
                        pseudo_text=data.get("pseudo_text", ""),
                    )
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> UnifiedQADatum:
        return self.samples[idx]


class UnifiedQACollator:
    """Collator that loads images and tokenizes text."""

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = 256,
        image_root: str = "",
        image_size: int = 448,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_root = image_root
        self.image_size = image_size
        self._load_failures = 0

    def _resolve_path(self, path: str) -> str:
        if self.image_root and not os.path.isabs(path):
            return os.path.join(self.image_root, path)
        return path

    def _load_image(self, path: str) -> Image.Image:
        try:
            image = Image.open(self._resolve_path(path)).convert("RGB")
            if self.image_size:
                image = image.resize((self.image_size, self.image_size))
            return image
        except (FileNotFoundError, OSError, ValueError) as exc:
            self._load_failures += 1
            if self._load_failures <= 10:
                logging.warning("Failed to load image %s: %s", path, exc)
            elif self._load_failures == 11:
                logging.warning("Suppressing further image load warnings.")
            size = self.image_size or 448
            return Image.new("RGB", (size, size), (0, 0, 0))

    def _tokenize(self, texts: List[str]) -> Optional[Dict[str, torch.Tensor]]:
        if self.tokenizer is None:
            return None
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, batch: List[UnifiedQADatum]) -> Dict[str, Any]:
        images = [self._load_image(item.image_path) for item in batch]
        image_paths = [item.image_path for item in batch]
        questions = [item.question for item in batch]
        answers = [item.answer for item in batch]
        pseudo_texts = [item.pseudo_text for item in batch]

        tokenized_questions = self._tokenize(questions)
        tokenized_pseudo = self._tokenize(pseudo_texts)

        return {
            "clean": {
                "images": images,
                "image_paths": image_paths,
                "questions": questions,
                "answers": answers,
                "tokenized_questions": tokenized_questions,
            },
            "corrupted": {
                "images": images,
                "image_paths": image_paths,
                "questions": questions,
                "pseudo_texts": pseudo_texts,
                "tokenized_pseudo_texts": tokenized_pseudo,
            },
        }
