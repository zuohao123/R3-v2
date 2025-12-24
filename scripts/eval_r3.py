"""Evaluate a trained R3++ checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from config.train_config import TrainConfig
from evaluation.evaluate import evaluate_model, save_results
from models.qwen_wrapper import QwenVLConfig, QwenVLWrapper
from models.r3_modules import R3PlusPlus
from retrieval.image_retrieval import ImageRetriever
from retrieval.text_retrieval import TextRetriever
import torch
from data.datasets import UnifiedQACollator, UnifiedQADataset
from torch.utils.data import DataLoader


def _update_dataclass(cfg: Any, updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if hasattr(cfg, key):
            sub = getattr(cfg, key)
            if isinstance(value, dict) and hasattr(sub, "__dict__"):
                _update_dataclass(sub, value)
            else:
                setattr(cfg, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate R3++ checkpoint")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--index_dir", default="indices")
    parser.add_argument("--out_json", default="eval_results.json")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--model_name",
        default=None,
        help="Override model path or HF id for evaluation",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _update_dataclass(cfg, data)

    cfg.data.val_jsonl = args.val_jsonl
    cfg.data.image_root = args.image_root
    cfg.retrieval.index_dir = args.index_dir
    cfg.retrieval.image_index_path = f"{args.index_dir}/image.index"
    cfg.retrieval.image_meta_path = f"{args.index_dir}/image.meta.json"
    cfg.retrieval.image_embeds_path = f"{args.index_dir}/image.embeds.npy"
    cfg.retrieval.text_index_path = f"{args.index_dir}/text.index"
    cfg.retrieval.text_meta_path = f"{args.index_dir}/text.meta.json"
    cfg.retrieval.text_embeds_path = f"{args.index_dir}/text.embeds.npy"
    cfg.retrieval.top_k = args.top_k

    if args.model_name:
        cfg.model.model_name = args.model_name
    else:
        cfg.model.model_name = args.checkpoint_dir

    qwen_cfg = QwenVLConfig(
        model_name=cfg.model.model_name,
        torch_dtype=cfg.model.torch_dtype,
        device=cfg.model.device,
        use_lora=cfg.model.use_lora,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        lora_target_modules=cfg.model.lora_target_modules,
    )
    qwen = QwenVLWrapper(qwen_cfg)

    text_retriever = TextRetriever(cfg.retrieval.text_encoder_name)
    image_retriever = ImageRetriever(cfg.retrieval.image_encoder_name, device=cfg.model.device)
    text_retriever.load(
        cfg.retrieval.text_index_path,
        cfg.retrieval.text_meta_path,
        cfg.retrieval.text_embeds_path,
    )
    image_retriever.load(
        cfg.retrieval.image_index_path,
        cfg.retrieval.image_meta_path,
        cfg.retrieval.image_embeds_path,
    )

    r3 = R3PlusPlus(qwen, text_retriever, image_retriever, cfg.r3)
    r3_state = os.path.join(args.checkpoint_dir, "r3_state.pt")
    if os.path.exists(r3_state):
        r3.load_state_dict(
            torch.load(r3_state, map_location=qwen.device),
            strict=False,
        )

    dataset = UnifiedQADataset(cfg.data.val_jsonl, image_root=cfg.data.image_root)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        collate_fn=UnifiedQACollator(
            tokenizer=qwen.tokenizer,
            max_length=cfg.data.max_length,
            image_root=cfg.data.image_root,
            image_size=cfg.data.image_size,
        ),
    )

    results = evaluate_model(
        r3,
        dataloader,
        corruption_levels=[0.0, 0.2, 0.4, 0.6, 0.8],
        max_new_tokens=cfg.evaluation.max_new_tokens,
        top_k=cfg.retrieval.top_k,
    )
    save_results(results, args.out_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
