"""Build FAISS indices for image and text retrieval (single or multi-GPU)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image
import torch

from retrieval.image_retrieval import ImageRetriever
from retrieval.text_retrieval import TextRetriever
from retrieval.vector_store import FaissVectorStore


def _load_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _resolve_image_path(path: str, image_root: str) -> str:
    if image_root and not os.path.isabs(path):
        return os.path.join(image_root, path)
    return path


def _filter_shard(records: List[Dict[str, Any]], shard_id: int, num_shards: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return records
    return [record for idx, record in enumerate(records) if idx % num_shards == shard_id]


def _get_shard_id(shard_id: Optional[int]) -> int:
    if shard_id is not None:
        return shard_id
    env_rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
    if env_rank is None:
        raise ValueError("--shard_id is required when --num_shards > 1.")
    return int(env_rank)


def _collect_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_path": record.get("image_path", ""),
        "pseudo_text": record.get("pseudo_text", ""),
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
    }


def _build_shard(
    records: List[Dict[str, Any]],
    image_root: str,
    out_dir: str,
    image_encoder: str,
    text_encoder: str,
    batch_size: int,
    device: Optional[str],
    shard_id: int,
    log_every: int,
) -> None:
    shard_dir = os.path.join(out_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Building shard %d with %d records on %s", shard_id, len(records), device)

    # Text embeddings
    text_retriever = TextRetriever(text_encoder, device=device)
    text_records = records
    text_meta = [_collect_metadata(r) for r in text_records]
    if text_records:
        texts = [r.get("pseudo_text", "") for r in text_records]
        text_embeds = text_retriever.encode_texts(texts, batch_size=batch_size)
        text_embeds_path = os.path.join(shard_dir, f"text.embeds.{shard_id}.npy")
        text_meta_path = os.path.join(shard_dir, f"text.meta.{shard_id}.json")
        np.save(text_embeds_path, text_embeds)
        with open(text_meta_path, "w", encoding="utf-8") as f:
            json.dump(text_meta, f)
        logging.info("Saved text shard %d: %s", shard_id, text_embeds_path)
    else:
        logging.warning("Shard %d has no text records; skipping text embeddings", shard_id)

    # Image embeddings
    image_retriever = ImageRetriever(image_encoder, device=device)
    image_meta: List[Dict[str, Any]] = []
    image_embeds_list: List[np.ndarray] = []
    images: List[Image.Image] = []

    for idx, record in enumerate(records):
        img_path = _resolve_image_path(record.get("image_path", ""), image_root)
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            continue
        images.append(image)
        image_meta.append(_collect_metadata(record))
        if len(images) >= batch_size:
            embeds = image_retriever.encode_images(images)
            image_embeds_list.append(embeds)
            images = []
        if log_every > 0 and (idx + 1) % log_every == 0:
            logging.info("Shard %d image progress: %d/%d", shard_id, idx + 1, len(records))

    if images:
        embeds = image_retriever.encode_images(images)
        image_embeds_list.append(embeds)

    if image_embeds_list:
        image_embeds = np.concatenate(image_embeds_list, axis=0)
        image_embeds_path = os.path.join(shard_dir, f"image.embeds.{shard_id}.npy")
        image_meta_path = os.path.join(shard_dir, f"image.meta.{shard_id}.json")
        np.save(image_embeds_path, image_embeds)
        with open(image_meta_path, "w", encoding="utf-8") as f:
            json.dump(image_meta, f)
        logging.info("Saved image shard %d: %s", shard_id, image_embeds_path)
    else:
        logging.warning("Shard %d has no images; skipping image embeddings", shard_id)


def _merge_shards(out_dir: str, num_shards: int) -> None:
    shard_dir = os.path.join(out_dir, "shards")

    def _merge(kind: str) -> None:
        embeds_list: List[np.ndarray] = []
        meta: List[Dict[str, Any]] = []
        for shard_id in range(num_shards):
            embeds_path = os.path.join(shard_dir, f"{kind}.embeds.{shard_id}.npy")
            meta_path = os.path.join(shard_dir, f"{kind}.meta.{shard_id}.json")
            if not os.path.exists(embeds_path) or not os.path.exists(meta_path):
                logging.warning("Missing shard %s %d, skipping", kind, shard_id)
                continue
            embeds_list.append(np.load(embeds_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                meta.extend(json.load(f))
        if not embeds_list:
            logging.warning("No %s shards found; skipping merge", kind)
            return
        embeds = np.concatenate(embeds_list, axis=0)
        store = FaissVectorStore(dim=embeds.shape[1], normalize=True)
        store.add(embeds, meta)
        store.save(
            os.path.join(out_dir, f"{kind}.index"),
            os.path.join(out_dir, f"{kind}.meta.json"),
            os.path.join(out_dir, f"{kind}.embeds.npy"),
        )
        logging.info("Merged %s index with %d entries", kind, len(meta))

    _merge("image")
    _merge("text")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build image/text retrieval indices.")
    parser.add_argument("--jsonl", nargs="+", required=False, help="Unified JSONL paths")
    parser.add_argument("--image_root", default="", help="Optional image root")
    parser.add_argument("--out_dir", default="indices", help="Output directory for indices")
    parser.add_argument("--image_encoder", default="models/clip-vit-b32-laion2B")
    parser.add_argument("--text_encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=None)
    parser.add_argument("--merge_shards", type=int, default=None)
    parser.add_argument("--device", default=None, help="Device for encoding, e.g. cuda:0")
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--write_combined", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge_shards:
        _merge_shards(args.out_dir, args.merge_shards)
        return

    if not args.jsonl:
        raise ValueError("--jsonl is required unless --merge_shards is set")

    records = _load_jsonl(args.jsonl)
    if args.write_combined or args.num_shards <= 1:
        combined_path = os.path.join(args.out_dir, "combined.jsonl")
        _write_jsonl(records, combined_path)

    if args.num_shards > 1:
        shard_id = _get_shard_id(args.shard_id)
        shard_records = _filter_shard(records, shard_id, args.num_shards)
        _build_shard(
            shard_records,
            image_root=args.image_root,
            out_dir=args.out_dir,
            image_encoder=args.image_encoder,
            text_encoder=args.text_encoder,
            batch_size=args.batch_size,
            device=args.device,
            shard_id=shard_id,
            log_every=args.log_every,
        )
        return

    default_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_retriever = ImageRetriever(args.image_encoder, device=default_device)
    image_retriever.build_index(
        os.path.join(args.out_dir, "combined.jsonl"),
        image_root=args.image_root,
        index_path=os.path.join(args.out_dir, "image.index"),
        meta_path=os.path.join(args.out_dir, "image.meta.json"),
        embeds_path=os.path.join(args.out_dir, "image.embeds.npy"),
        batch_size=args.batch_size,
    )

    text_retriever = TextRetriever(args.text_encoder, device=default_device)
    text_retriever.build_index(
        os.path.join(args.out_dir, "combined.jsonl"),
        index_path=os.path.join(args.out_dir, "text.index"),
        meta_path=os.path.join(args.out_dir, "text.meta.json"),
        embeds_path=os.path.join(args.out_dir, "text.embeds.npy"),
    )

    logging.info("Indices saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
