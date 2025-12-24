"""Download retrieval backbone models to a local models directory."""
from __future__ import annotations

import argparse
import logging
import os

from huggingface_hub import snapshot_download


def _download(repo_id: str, out_dir: str, name: str) -> str:
    target_dir = os.path.join(out_dir, name)
    os.makedirs(target_dir, exist_ok=True)
    logging.info("Downloading %s to %s", repo_id, target_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download retrieval models.")
    parser.add_argument("--out_dir", default="models", help="Local models directory")
    parser.add_argument(
        "--clip_model", default="openai/clip-vit-base-patch32", help="CLIP repo id"
    )
    parser.add_argument(
        "--text_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Text encoder repo id",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    _download(args.clip_model, args.out_dir, "clip-vit-base-patch32")
    _download(args.text_model, args.out_dir, "all-MiniLM-L6-v2")


if __name__ == "__main__":
    main()
