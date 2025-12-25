"""Text retrieval using sentence-transformers + FAISS."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np

_OCR_RE = re.compile(r"(?:^|\\b)OCR\\s*:\\s*(.*)$", re.IGNORECASE)


def _extract_ocr_text(record: Dict[str, Any]) -> str:
    text = str(record.get("ocr_text", "") or "").strip()
    if text:
        return text
    pseudo = str(record.get("pseudo_text", "") or "").strip()
    if not pseudo:
        return ""
    match = _OCR_RE.search(pseudo)
    return match.group(1).strip() if match else ""


def _select_text(record: Dict[str, Any], text_field: str) -> str:
    field = text_field.lower()
    if field == "ocr":
        return _extract_ocr_text(record)
    if field == "question":
        return str(record.get("question", "") or "").strip()
    if field == "pseudo_text":
        return str(record.get("pseudo_text", "") or "").strip()
    text = _extract_ocr_text(record)
    if text:
        return text
    text = str(record.get("pseudo_text", "") or "").strip()
    if text:
        return text
    return str(record.get("question", "") or "").strip()

from retrieval.vector_store import FaissVectorStore


class TextRetriever:
    """Transformer-based text retriever."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
        self.store: Optional[FaissVectorStore] = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        from sentence_transformers import SentenceTransformer

        if self.device:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        else:
            self.model = SentenceTransformer(self.model_name)

    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._ensure_model()
        assert self.model is not None
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )

    def build_index(
        self,
        jsonl_path: str,
        index_path: str,
        meta_path: str,
        embeds_path: str,
        max_samples: Optional[int] = None,
        text_field: str = "auto",
    ) -> None:
        self._ensure_model()
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                if max_samples is not None and len(records) >= max_samples:
                    break

        texts = [_select_text(record, text_field) for record in records]
        embeddings = self.encode_texts(texts)
        metadata = [
            {
                "image_path": record.get("image_path", ""),
                "pseudo_text": text,
                "full_pseudo_text": record.get("pseudo_text", ""),
                "ocr_text": record.get("ocr_text", ""),
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
            }
            for record, text in zip(records, texts)
        ]
        store = FaissVectorStore(dim=embeddings.shape[1], normalize=True)
        store.add(embeddings, metadata)
        store.save(index_path, meta_path, embeds_path)
        self.store = store

    def load(self, index_path: str, meta_path: str, embeds_path: Optional[str]) -> None:
        self.store = FaissVectorStore.load(index_path, meta_path, embeds_path, normalize=True)

    def retrieve(self, texts: List[str], top_k: int) -> Dict[str, Any]:
        if self.store is None:
            raise RuntimeError("Text index not loaded.")
        query = self.encode_texts(texts)
        scores, metas, embeds = self.store.search(query, top_k)
        return {"scores": scores, "metadata": metas, "embeddings": embeds}
