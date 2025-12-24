"""Image retrieval using CLIP + FAISS."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch

from retrieval.vector_store import FaissVectorStore


class ImageRetriever:
    """CLIP-based image retriever."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.store: Optional[FaissVectorStore] = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        self._ensure_model()
        assert self.processor is not None
        assert self.model is not None
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy()

    def build_index(
        self,
        jsonl_path: str,
        image_root: str,
        index_path: str,
        meta_path: str,
        embeds_path: str,
        batch_size: int = 16,
        max_samples: Optional[int] = None,
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

        def _resolve(path: str) -> str:
            return path if os.path.isabs(path) or not image_root else os.path.join(image_root, path)

        images: List[Image.Image] = []
        metadata: List[Dict[str, Any]] = []
        all_embeddings: List[np.ndarray] = []

        for record in records:
            image_path = _resolve(record["image_path"])
            try:
                image = Image.open(image_path).convert("RGB")
            except (FileNotFoundError, OSError):
                continue
            images.append(image)
            metadata.append(
                {
                    "image_path": record["image_path"],
                    "pseudo_text": record.get("pseudo_text", ""),
                    "question": record.get("question", ""),
                    "answer": record.get("answer", ""),
                }
            )
            if len(images) >= batch_size:
                embeddings = self.encode_images(images)
                all_embeddings.append(embeddings)
                images = []
        if images:
            embeddings = self.encode_images(images)
            all_embeddings.append(embeddings)

        if not all_embeddings:
            raise RuntimeError("No images were encoded for the index.")
        embeddings = np.concatenate(all_embeddings, axis=0)
        store = FaissVectorStore(dim=embeddings.shape[1], normalize=True)
        store.add(embeddings, metadata)
        store.save(index_path, meta_path, embeds_path)
        self.store = store

    def load(self, index_path: str, meta_path: str, embeds_path: Optional[str]) -> None:
        self.store = FaissVectorStore.load(index_path, meta_path, embeds_path, normalize=True)

    def retrieve(self, images: List[Image.Image], top_k: int) -> Dict[str, Any]:
        if self.store is None:
            raise RuntimeError("Image index not loaded.")
        query = self.encode_images(images)
        scores, metas, embeds = self.store.search(query, top_k)
        return {"scores": scores, "metadata": metas, "embeddings": embeds}
