#!/usr/bin/env python3
"""Compute oracle-match between heuristic routing and best-of routing."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset

from config.train_config import TrainConfig
from data.datasets import UnifiedQACollator, UnifiedQADataset, UnifiedQADatum
from evaluation import evaluate as eval_utils
from models.qwen_wrapper import QwenVLConfig, QwenVLWrapper
from models.r3_modules import R3
from retrieval.image_retrieval import ImageRetriever
from retrieval.text_retrieval import TextRetriever


class _ListQADataset(Dataset):
    def __init__(self, samples: List[UnifiedQADatum]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> UnifiedQADatum:
        return self.samples[idx]


class _FixedCorruptor(torch.nn.Module):
    def __init__(self, images, texts, c_vis, c_text) -> None:
        super().__init__()
        self.images = images
        self.texts = texts
        self.c_vis = c_vis
        self.c_text = c_text

    def forward(self, *_args, **_kwargs):
        return self.images, self.texts, self.c_vis, self.c_text

    __call__ = forward


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


def _is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _init_distributed() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _shard_dataset(dataset: Dataset, rank: int, world_size: int) -> Dataset:
    if world_size <= 1:
        return dataset
    indices = list(range(rank, len(dataset), world_size))
    return Subset(dataset, indices)


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
                ocr_conf=float(data.get("ocr_conf", 1.0)),
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


def _choice_from_alpha(alpha: torch.Tensor) -> List[int]:
    if alpha.dim() == 1:
        alpha = alpha.unsqueeze(1)
    if alpha.size(1) == 1:
        alpha_prefix = alpha[:, 0]
        alpha_latent = alpha[:, 0]
    else:
        alpha_prefix = alpha[:, 0]
        alpha_latent = alpha[:, 1]
    choices: List[int] = []
    for prefix_val, latent_val in zip(alpha_prefix.tolist(), alpha_latent.tolist()):
        if prefix_val <= 1e-6:
            choices.append(0)  # retrieval-only
        elif latent_val > 0.5:
            choices.append(2)  # reconstruct
        else:
            choices.append(1)  # hybrid/prefix-only
    return choices


def _reduce_counts(counts: Dict[Tuple[str, float], Tuple[int, int, int]], device: str) -> Dict:
    if not dist.is_initialized():
        return counts
    reduced: Dict[Tuple[str, float], Tuple[int, int, int]] = {}
    for key, (match, oracle_total, total) in counts.items():
        tensor = torch.tensor([match, oracle_total, total], device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = (int(tensor[0].item()), int(tensor[1].item()), int(tensor[2].item()))
    return reduced


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Compute oracle-match for routing.")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--index_dir", default="indices")
    parser.add_argument("--out_json", default="oracle_match.json")
    parser.add_argument("--dataset_prefixes", default="screenqa,chartqa,infovqa")
    parser.add_argument("--corruption_levels", default="0.6,0.8,1.0")
    parser.add_argument("--disable_corruption", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_context_chars", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--answer_only", action="store_true")
    parser.add_argument("--no_pseudo_text", action="store_true")
    parser.add_argument("--heuristic_mode", default="heuristic_retrieval")
    parser.add_argument("--router_options", default="0,0;1,0;1,1")
    parser.add_argument("--skip_no_oracle", action="store_true")
    args = parser.parse_args()

    distributed = _is_distributed()
    rank = 0
    world_size = 1
    local_rank = 0
    if distributed:
        rank, world_size, local_rank = _init_distributed()
    is_main_process = rank == 0

    cfg = TrainConfig()
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _update_dataclass(cfg, data)

    cfg.data.val_jsonl = args.val_jsonl
    cfg.data.image_root = args.image_root
    cfg.data.max_length = args.max_length
    cfg.evaluation.max_eval_samples = args.max_eval_samples
    cfg.evaluation.batch_size = args.batch_size
    cfg.evaluation.max_new_tokens = args.max_new_tokens
    cfg.r3.max_context_chars = args.max_context_chars
    cfg.retrieval.top_k = args.top_k
    cfg.r3.enable_corruption = not args.disable_corruption

    if distributed:
        cfg.model.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

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
    adapter_safetensors = os.path.join(args.checkpoint_dir, "adapter_model.safetensors")
    adapter_bin = os.path.join(args.checkpoint_dir, "adapter_model.bin")
    if os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin):
        qwen.load_lora_adapter(args.checkpoint_dir)

    cfg.retrieval.index_dir = args.index_dir
    cfg.retrieval.text_index_path = f"{args.index_dir}/text.index"
    cfg.retrieval.text_meta_path = f"{args.index_dir}/text.meta.json"
    cfg.retrieval.text_embeds_path = f"{args.index_dir}/text.embeds.npy"
    cfg.retrieval.image_index_path = f"{args.index_dir}/image.index"
    cfg.retrieval.image_meta_path = f"{args.index_dir}/image.meta.json"
    cfg.retrieval.image_embeds_path = f"{args.index_dir}/image.embeds.npy"

    text_retriever = None
    image_retriever = None
    if os.path.exists(cfg.retrieval.text_index_path) and os.path.exists(cfg.retrieval.text_meta_path):
        text_retriever = TextRetriever(cfg.retrieval.text_encoder_name)
        text_retriever.load(
            cfg.retrieval.text_index_path,
            cfg.retrieval.text_meta_path,
            cfg.retrieval.text_embeds_path,
        )
    if os.path.exists(cfg.retrieval.image_index_path) and os.path.exists(cfg.retrieval.image_meta_path):
        image_retriever = ImageRetriever(cfg.retrieval.image_encoder_name, device=cfg.model.device)
        image_retriever.load(
            cfg.retrieval.image_index_path,
            cfg.retrieval.image_meta_path,
            cfg.retrieval.image_embeds_path,
        )

    r3 = R3(qwen, text_retriever, image_retriever, cfg.r3)
    r3_state = os.path.join(args.checkpoint_dir, "r3_state.pt")
    if os.path.exists(r3_state):
        r3.load_state_dict(torch.load(r3_state, map_location=qwen.device), strict=False)
    r3.to(qwen.device)
    r3.eval()

    levels = _parse_levels(args.corruption_levels)
    prefixes = [p.strip() for p in args.dataset_prefixes.split(",") if p.strip()]
    grouped = _load_grouped_samples(args.val_jsonl, prefixes, args.max_eval_samples)

    router_options: List[Tuple[float, float]] = []
    for token in args.router_options.split(";"):
        parts = [p.strip() for p in token.split(",") if p.strip()]
        if len(parts) == 1:
            router_options.append((float(parts[0]), float(parts[0])))
        else:
            router_options.append((float(parts[0]), float(parts[1])))

    counts: Dict[Tuple[str, float], Tuple[int, int, int]] = {}
    with torch.no_grad():
        for prefix in prefixes:
            samples = grouped.get(prefix, [])
            if is_main_process:
                logging.info("Dataset=%s | samples=%d", prefix, len(samples))
            subset = _ListQADataset(samples)
            subset = _shard_dataset(subset, rank, world_size)
            loader = DataLoader(
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
            for level in levels:
                match = 0
                oracle_total = 0
                total = 0
                for batch in loader:
                    clean = batch["clean"]
                    corrupted = batch["corrupted"]
                    images = corrupted["images"]
                    questions = corrupted["questions"]
                    pseudo_texts = corrupted["pseudo_texts"]
                    ocr_confs = corrupted.get("ocr_confs")
                    refs = clean["answers"]
                    batch_size = len(questions)
                    if args.no_pseudo_text:
                        pseudo_texts = ["" for _ in questions]

                    if cfg.r3.enable_corruption and level > 0.0:
                        corr_images, corr_texts, c_vis, c_text = r3.corruptor(
                            images, pseudo_texts, level
                        )
                    else:
                        corr_images = images
                        corr_texts = pseudo_texts
                        c_vis = torch.ones(batch_size, 1)
                        c_text = torch.ones(batch_size, 1)

                    queries = [f"{q} {t}" for q, t in zip(questions, corr_texts)]
                    text_embeds, retrieved_texts, text_scores = r3._retrieve_texts(queries, cfg.retrieval.top_k)
                    image_embeds, retrieved_images, retrieved_paths, image_scores = r3._retrieve_images(
                        corr_images, cfg.retrieval.top_k
                    )
                    (
                        text_embeds,
                        retrieved_texts,
                        text_scores,
                        image_embeds,
                        retrieved_images,
                        retrieved_paths,
                        image_scores,
                    ) = r3._maybe_shuffle_retrieval(
                        text_embeds,
                        retrieved_texts,
                        text_scores,
                        image_embeds,
                        retrieved_images,
                        retrieved_paths,
                        image_scores,
                    )
                    c_vis_adj, c_text_adj, _ = r3._apply_retrieval_confidence(
                        c_vis.to(r3.qwen.device),
                        c_text.to(r3.qwen.device),
                        text_scores,
                        image_scores,
                        ocr_confs,
                    )
                    heur_alpha = r3._heuristic_router_alpha(
                        args.heuristic_mode,
                        c_vis_adj,
                        c_text_adj,
                        text_scores,
                        image_scores,
                        r3.config.router_out_dim,
                    )
                    heur_choices = _choice_from_alpha(heur_alpha)

                    fixed = _FixedCorruptor(corr_images, corr_texts, c_vis, c_text)
                    orig_corruptor = r3.corruptor
                    r3.corruptor = fixed
                    preds_by: List[List[str]] = []
                    for override in router_options:
                        preds = r3.generate(
                            images,
                            questions,
                            pseudo_texts,
                            corruption_level=level,
                            top_k=cfg.retrieval.top_k,
                            max_new_tokens=cfg.evaluation.max_new_tokens,
                            answer_only=args.answer_only,
                            ocr_confs=ocr_confs,
                            router_alpha_override=override,
                        )
                        preds_by.append(preds)
                    r3.corruptor = orig_corruptor

                    for i in range(batch_size):
                        oracle_set = []
                        for opt_idx, preds in enumerate(preds_by):
                            metrics = eval_utils._compute_metrics(preds[i], refs[i])
                            if metrics.get("exact_match", 0.0) > 0.5:
                                oracle_set.append(opt_idx)
                        total += 1
                        if oracle_set:
                            oracle_total += 1
                            if heur_choices[i] in oracle_set:
                                match += 1
                        elif not args.skip_no_oracle:
                            if heur_choices[i] in oracle_set:
                                match += 1

                counts[(prefix, level)] = (match, oracle_total, total)

    device = cfg.model.device if isinstance(cfg.model.device, str) else "cpu"
    counts = _reduce_counts(counts, device)
    if is_main_process:
        result = {}
        for (prefix, level), (match, oracle_total, total) in counts.items():
            denom = oracle_total if args.skip_no_oracle else total
            rate = match / max(1, denom)
            result.setdefault(prefix, {})[level] = {
                "oracle_match": round(rate * 100.0, 2),
                "match": match,
                "oracle_total": oracle_total,
                "total": total,
                "skip_no_oracle": bool(args.skip_no_oracle),
            }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logging.info("Saved oracle-match results to %s", args.out_json)
        logging.info("%s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
