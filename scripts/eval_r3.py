"""Evaluate a trained R3 checkpoint or base model."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.train_config import TrainConfig
from evaluation.evaluate import evaluate_model, save_results
from models.qwen_wrapper import QwenVLConfig, QwenVLWrapper
from models.r3_modules import CorruptionSimulator, R3
from retrieval.image_retrieval import ImageRetriever
from retrieval.text_retrieval import TextRetriever
import torch
from data.datasets import UnifiedQACollator, UnifiedQADataset, UnifiedQADatum
from torch.utils.data import DataLoader, Dataset


def _update_dataclass(cfg: Any, updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if hasattr(cfg, key):
            sub = getattr(cfg, key)
            if isinstance(value, dict) and hasattr(sub, "__dict__"):
                _update_dataclass(sub, value)
            else:
                setattr(cfg, key, value)


def _parse_levels(value: str) -> List[float]:
    return [float(x) for x in value.split(",") if x.strip()]

class _ListQADataset(Dataset):
    def __init__(self, samples: List[UnifiedQADatum]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> UnifiedQADatum:
        return self.samples[idx]


def _match_prefix(path: str, prefix: str) -> bool:
    prefix = prefix.strip().strip("/")
    if not prefix:
        return False
    norm = path.replace("\\", "/").lstrip("./")
    if norm.startswith(prefix + "/"):
        return True
    parts = [p for p in norm.split("/") if p]
    return prefix in parts


def _load_grouped_samples(
    jsonl_path: str, prefixes: List[str], max_samples: Optional[int]
) -> Dict[str, List[UnifiedQADatum]]:
    groups: Dict[str, List[UnifiedQADatum]] = {p: [] for p in prefixes}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            datum = UnifiedQADatum(
                image_path=data["image_path"],
                question=data["question"],
                answer=data["answer"],
                pseudo_text=data.get("pseudo_text", ""),
            )
            matched = False
            for prefix in prefixes:
                if _match_prefix(datum.image_path, prefix):
                    groups[prefix].append(datum)
                    matched = True
                    break
            if matched and max_samples is not None:
                if all(len(groups[p]) >= max_samples for p in prefixes):
                    for prefix in prefixes:
                        groups[prefix] = groups[prefix][:max_samples]
                    break
    return groups


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate R3 or base model")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--index_dir", default="indices")
    parser.add_argument("--out_json", default="eval_results.json")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Evaluate a single dataset prefix (e.g., screenqa, chartqa, infovqa).",
    )
    parser.add_argument(
        "--dataset_prefixes",
        default=None,
        help="Comma-separated dataset prefixes to evaluate separately.",
    )
    parser.add_argument("--eval_log_every", type=int, default=None)
    parser.add_argument(
        "--eval_mode",
        choices=["r3", "base"],
        default="r3",
        help="Evaluate R3 model or base Qwen model",
    )
    parser.add_argument(
        "--corruption_levels",
        default="0.0,0.2,0.4,0.6,0.8",
        help="Comma-separated corruption levels",
    )
    parser.add_argument("--clean_only", action="store_true")
    parser.add_argument("--use_pseudo_text", action="store_true")
    parser.add_argument("--no_pseudo_text", action="store_true")
    parser.add_argument(
        "--corrupt_text_target",
        choices=["pseudo_text", "question", "none"],
        default="pseudo_text",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Override model path or HF id for evaluation",
    )
    parser.add_argument("--disable_corruption", action="store_true")
    parser.add_argument("--disable_text_retrieval", action="store_true")
    parser.add_argument("--disable_image_retrieval", action="store_true")
    parser.add_argument("--disable_prefix", action="store_true")
    parser.add_argument("--disable_memory", action="store_true")
    parser.add_argument("--disable_visual_memory", action="store_true")
    parser.add_argument("--disable_latent_tokens", action="store_true")
    parser.add_argument("--disable_gate", action="store_true")
    parser.add_argument("--disable_context", action="store_true")
    parser.add_argument("--use_soft_prefix", action="store_true")
    parser.add_argument("--disable_soft_prefix", action="store_true")
    parser.add_argument(
        "--load_lora_adapter",
        action="store_true",
        help="Load LoRA adapter weights from --checkpoint_dir if present.",
    )
    parser.add_argument("--max_context_chars", type=int, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.checkpoint_dir:
        config_path = os.path.join(args.checkpoint_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _update_dataclass(cfg, data)

    cfg.data.val_jsonl = args.val_jsonl
    cfg.data.image_root = args.image_root
    if args.max_length is not None:
        cfg.data.max_length = args.max_length
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.max_eval_samples is not None:
        cfg.evaluation.max_eval_samples = args.max_eval_samples
    cfg.retrieval.index_dir = args.index_dir
    cfg.retrieval.image_index_path = f"{args.index_dir}/image.index"
    cfg.retrieval.image_meta_path = f"{args.index_dir}/image.meta.json"
    cfg.retrieval.image_embeds_path = f"{args.index_dir}/image.embeds.npy"
    cfg.retrieval.text_index_path = f"{args.index_dir}/text.index"
    cfg.retrieval.text_meta_path = f"{args.index_dir}/text.meta.json"
    cfg.retrieval.text_embeds_path = f"{args.index_dir}/text.embeds.npy"
    cfg.retrieval.top_k = args.top_k
    cfg.r3.enable_corruption = not args.disable_corruption
    cfg.r3.enable_text_retrieval = not args.disable_text_retrieval
    cfg.r3.enable_image_retrieval = not args.disable_image_retrieval
    cfg.r3.enable_prefix = not args.disable_prefix
    cfg.r3.enable_memory = not args.disable_memory
    cfg.r3.enable_visual_memory = not args.disable_visual_memory
    cfg.r3.enable_latent_tokens = not args.disable_latent_tokens
    cfg.r3.enable_gate = not args.disable_gate
    cfg.r3.enable_context = not args.disable_context
    if args.max_context_chars is not None:
        cfg.r3.max_context_chars = args.max_context_chars
    if args.use_soft_prefix:
        cfg.r3.use_soft_prefix = True
    if args.disable_soft_prefix:
        cfg.r3.use_soft_prefix = False

    if args.model_name:
        cfg.model.model_name = args.model_name
    elif args.checkpoint_dir and not cfg.model.model_name:
        cfg.model.model_name = args.checkpoint_dir

    if args.eval_mode == "r3" and not cfg.model.model_name:
        raise ValueError("--checkpoint_dir or --model_name is required for eval_mode=r3")

    adapter_path = None
    if args.checkpoint_dir:
        adapter_safetensors = os.path.join(args.checkpoint_dir, "adapter_model.safetensors")
        adapter_bin = os.path.join(args.checkpoint_dir, "adapter_model.bin")
        if os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin):
            adapter_path = args.checkpoint_dir
            if args.eval_mode == "r3" or args.load_lora_adapter:
                cfg.model.use_lora = True

    qwen_cfg = QwenVLConfig(
        model_name=cfg.model.model_name,
        torch_dtype=cfg.model.torch_dtype,
        device=cfg.model.device,
        use_teacher=False,
        use_lora=cfg.model.use_lora,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        lora_target_modules=cfg.model.lora_target_modules,
    )
    qwen = QwenVLWrapper(qwen_cfg)
    if adapter_path and cfg.model.use_lora:
        qwen.load_lora_adapter(adapter_path)
    r3 = None
    if args.eval_mode == "r3":
        text_retriever = None
        image_retriever = None
        if cfg.r3.enable_text_retrieval:
            text_retriever = TextRetriever(cfg.retrieval.text_encoder_name)
            text_retriever.load(
                cfg.retrieval.text_index_path,
                cfg.retrieval.text_meta_path,
                cfg.retrieval.text_embeds_path,
            )
        if cfg.r3.enable_image_retrieval:
            image_retriever = ImageRetriever(
                cfg.retrieval.image_encoder_name, device=cfg.model.device
            )
            image_retriever.load(
                cfg.retrieval.image_index_path,
                cfg.retrieval.image_meta_path,
                cfg.retrieval.image_embeds_path,
            )

        r3 = R3(qwen, text_retriever, image_retriever, cfg.r3)
        if args.checkpoint_dir:
            r3_state = os.path.join(args.checkpoint_dir, "r3_state.pt")
            if os.path.exists(r3_state):
                r3.load_state_dict(
                    torch.load(r3_state, map_location=qwen.device),
                    strict=False,
                )

    dataset = UnifiedQADataset(
        cfg.data.val_jsonl,
        image_root=cfg.data.image_root,
        max_samples=cfg.evaluation.max_eval_samples,
    )
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

    if args.clean_only:
        levels = [0.0]
    else:
        levels = _parse_levels(args.corruption_levels)
    if args.dataset and args.dataset_prefixes:
        raise ValueError("Set either --dataset or --dataset_prefixes, not both.")
    dataset_prefixes = []
    if args.dataset:
        dataset_prefixes = [args.dataset]
    elif args.dataset_prefixes:
        dataset_prefixes = [p.strip() for p in args.dataset_prefixes.split(",") if p.strip()]
    use_pseudo_text: Optional[bool] = None
    if args.use_pseudo_text and args.no_pseudo_text:
        raise ValueError("Cannot set both --use_pseudo_text and --no_pseudo_text")
    if args.use_pseudo_text:
        use_pseudo_text = True
    if args.no_pseudo_text:
        use_pseudo_text = False

    corruptor = CorruptionSimulator(cfg.r3) if args.eval_mode == "base" else None
    model = r3 if args.eval_mode == "r3" else qwen
    if dataset_prefixes:
        grouped = _load_grouped_samples(
            cfg.data.val_jsonl, dataset_prefixes, cfg.evaluation.max_eval_samples
        )
        results_by_dataset: Dict[str, Dict[float, Dict[str, float]]] = {}
        for prefix in dataset_prefixes:
            samples = grouped.get(prefix, [])
            logging.info("Eval dataset=%s | samples=%d", prefix, len(samples))
            subset = _ListQADataset(samples)
            subset_loader = DataLoader(
                subset,
                batch_size=cfg.evaluation.batch_size,
                shuffle=False,
                collate_fn=UnifiedQACollator(
                    tokenizer=qwen.tokenizer,
                    max_length=cfg.data.max_length,
                    image_root=cfg.data.image_root,
                    image_size=cfg.data.image_size,
                ),
            )
            results_by_dataset[prefix] = evaluate_model(
                model,
                subset_loader,
                corruption_levels=levels,
                max_new_tokens=cfg.evaluation.max_new_tokens,
                top_k=cfg.retrieval.top_k,
                mode=args.eval_mode,
                use_pseudo_text=use_pseudo_text,
                corrupt_text_target=args.corrupt_text_target,
                corruptor=corruptor,
                log_every=args.eval_log_every,
            )
        save_results(results_by_dataset, args.out_json)
        print(json.dumps(results_by_dataset, indent=2))
        return

    results = evaluate_model(
        model,
        dataloader,
        corruption_levels=levels,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        top_k=cfg.retrieval.top_k,
        mode=args.eval_mode,
        use_pseudo_text=use_pseudo_text,
        corrupt_text_target=args.corrupt_text_target,
        corruptor=corruptor,
        log_every=args.eval_log_every,
    )
    save_results(results, args.out_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
