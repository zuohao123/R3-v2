"""Build FAISS indices for image and text retrieval."""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from retrieval.image_retrieval import ImageRetriever
from retrieval.text_retrieval import TextRetriever


def _load_jsonl(paths: List[str]) -> List[dict]:
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _write_jsonl(records: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build image/text retrieval indices.")
    parser.add_argument("--jsonl", nargs="+", required=True, help="Unified JSONL paths")
    parser.add_argument("--image_root", default="", help="Optional image root")
    parser.add_argument("--out_dir", default="indices", help="Output directory for indices")
    parser.add_argument("--image_encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--text_encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    records = _load_jsonl(args.jsonl)
    combined_path = os.path.join(args.out_dir, "combined.jsonl")
    _write_jsonl(records, combined_path)

    image_retriever = ImageRetriever(args.image_encoder)
    image_retriever.build_index(
        combined_path,
        image_root=args.image_root,
        index_path=os.path.join(args.out_dir, "image.index"),
        meta_path=os.path.join(args.out_dir, "image.meta.json"),
        embeds_path=os.path.join(args.out_dir, "image.embeds.npy"),
        batch_size=args.batch_size,
    )

    text_retriever = TextRetriever(args.text_encoder)
    text_retriever.build_index(
        combined_path,
        index_path=os.path.join(args.out_dir, "text.index"),
        meta_path=os.path.join(args.out_dir, "text.meta.json"),
        embeds_path=os.path.join(args.out_dir, "text.embeds.npy"),
    )

    print(f"Indices saved to {args.out_dir}")


if __name__ == "__main__":
    main()
