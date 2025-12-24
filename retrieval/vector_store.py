"""FAISS vector store for retrieval."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FaissVectorStore:
    """Thin wrapper over FAISS with metadata and optional embeddings."""

    def __init__(self, dim: int, normalize: bool = True) -> None:
        self.dim = dim
        self.normalize = normalize
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def _ensure_index(self) -> None:
        if self.index is not None:
            return
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("faiss is required for retrieval.") from exc
        if self.normalize:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexFlatL2(self.dim)

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        self._ensure_index()
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Embeddings must be (N, {self.dim})")
        if self.normalize:
            embeddings = self._normalize(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        self.embeddings = (
            embeddings
            if self.embeddings is None
            else np.concatenate([self.embeddings, embeddings], axis=0)
        )

    def search(
        self, query: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, List[List[Dict[str, Any]]], Optional[np.ndarray]]:
        self._ensure_index()
        if query.ndim == 1:
            query = query[None, :]
        if self.normalize:
            query = self._normalize(query)
        scores, indices = self.index.search(query, top_k)
        metas: List[List[Dict[str, Any]]] = []
        embeds: Optional[np.ndarray] = None
        for row in indices:
            row_meta = []
            for idx in row:
                if idx < 0:
                    row_meta.append({})
                else:
                    row_meta.append(self.metadata[idx])
            metas.append(row_meta)
        if self.embeddings is not None:
            embeds = self.embeddings[indices]
        return scores, metas, embeds

    def save(self, index_path: str, meta_path: str, embeds_path: Optional[str] = None) -> None:
        self._ensure_index()
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("faiss is required for retrieval.") from exc
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)
        if embeds_path and self.embeddings is not None:
            np.save(embeds_path, self.embeddings)

    @classmethod
    def load(
        cls,
        index_path: str,
        meta_path: str,
        embeds_path: Optional[str] = None,
        normalize: bool = True,
    ) -> "FaissVectorStore":
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError("faiss is required for retrieval.") from exc
        index = faiss.read_index(index_path)
        dim = index.d
        store = cls(dim=dim, normalize=normalize)
        store.index = index
        with open(meta_path, "r", encoding="utf-8") as f:
            store.metadata = json.load(f)
        if embeds_path and os.path.exists(embeds_path):
            store.embeddings = np.load(embeds_path)
        return store

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
        return embeddings / norm
