#!/usr/bin/env python
"""Analyze R3 gate weights across corruption levels without full generation."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from config.train_config import TrainConfig
from data.datasets import UnifiedQACollator, UnifiedQADataset, UnifiedQADatum
from models.qwen_wrapper import QwenVLConfig, QwenVLWrapper
from models.r3_modules import R3
from retrieval.image_retrieval import ImageRetriever
from retrieval.text_retrieval import TextRetriever


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


def _compute_gates_only(
    r3: R3,
    images: List[Any],
    questions: List[str],
    pseudo_texts: List[str],
    corruption_level: float,
    top_k: int,
    use_pseudo_text: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_pseudo_text:
        pseudo_texts = ["" for _ in questions]
    if r3.config.enable_corruption:
        corr_images, corr_texts, c_vis, c_text = r3.corruptor(
            images, pseudo_texts, corruption_level
        )
    else:
        corr_images = images
        corr_texts = pseudo_texts
        c_vis = torch.ones(len(images), 1)
        c_text = torch.ones(len(images), 1)

    queries = [f"{q} {t}" for q, t in zip(questions, corr_texts)]
    text_embeds, retrieved_texts, text_scores = r3._retrieve_texts(queries, top_k)
    image_embeds, _, _, image_scores = r3._retrieve_images(corr_images, top_k)

    text_embeds = text_embeds.to(r3.qwen.device, dtype=torch.float32)
    image_embeds = image_embeds.to(r3.qwen.device, dtype=torch.float32)
    c_vis = c_vis.to(r3.qwen.device, dtype=torch.float32)
    c_text = c_text.to(r3.qwen.device, dtype=torch.float32)

    text_embeds = r3._sanitize(text_embeds)
    image_embeds = r3._sanitize(image_embeds)
    text_embeds = F.normalize(text_embeds, dim=-1, eps=1e-6).detach()
    image_embeds = F.normalize(image_embeds, dim=-1, eps=1e-6).detach()

    if text_scores is not None:
        text_weights = r3._score_weights(text_scores, top_k, r3.config.min_text_score)
    else:
        text_weights = torch.full(
            (len(images), top_k), 1.0 / top_k, device=r3.qwen.device
        )
    if image_scores is not None:
        image_weights = r3._score_weights(image_scores, top_k, r3.config.min_image_score)
    else:
        image_weights = torch.full(
            (len(images), top_k), 1.0 / top_k, device=r3.qwen.device
        )

    if r3.memory_aligner is not None:
        mem_t, mem_i = r3.memory_aligner(
            text_embeds,
            image_embeds,
            text_weights=text_weights,
            image_weights=image_weights,
        )
        mem_t = mem_t * c_text
        mem_i = mem_i * c_vis
    else:
        mem_t = torch.zeros(
            (len(images), r3.config.hidden_dim),
            device=r3.qwen.device,
            dtype=torch.float32,
        )
        mem_i = torch.zeros(
            (len(images), r3.config.hidden_dim),
            device=r3.qwen.device,
            dtype=torch.float32,
        )

    mem_t = r3._sanitize(mem_t)
    mem_i = r3._sanitize(mem_i)
    mem_t_gate = mem_t.detach()
    mem_i_gate = mem_i.detach()
    vis_feat = r3.vis_proj(image_embeds.mean(dim=1))
    vis_feat = r3._sanitize(vis_feat)
    if r3.gate is not None:
        gates = r3.gate(mem_t_gate, mem_i_gate, vis_feat)
        gates = r3._sanitize(gates, fill=0.0)
        gates = torch.softmax(gates, dim=-1)
        gates = r3._sanitize(gates, fill=1.0 / 3)
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    else:
        gates = torch.full((len(images), 3), 1 / 3, device=r3.qwen.device)

    return gates, c_vis, c_text


def _reduce_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not dist.is_initialized():
        return stats
    reduced: Dict[str, torch.Tensor] = {}
    for key, value in stats.items():
        tensor = value.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[key] = tensor
    return reduced


def _finalize_stats(stats: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    count = max(1.0, float(stats["count"].item()))
    gate_sum = stats["gate_sum"].cpu().numpy().tolist()
    gate_sumsq = stats["gate_sumsq"].cpu().numpy().tolist()
    gate_mean = [v / count for v in gate_sum]
    gate_var = [max(0.0, (s / count) - (m * m)) for s, m in zip(gate_sumsq, gate_mean)]
    gate_std = [v ** 0.5 for v in gate_var]
    return {
        "count": int(stats["count"].item()),
        "gate_mean": gate_mean,
        "gate_std": gate_std,
        "gate_entropy": float(stats["gate_entropy"].item() / count),
        "retrieval_mass": float(stats["retrieval_mass"].item() / count),
        "latent_mass": float(stats["latent_mass"].item() / count),
        "n_text": float(stats["n_text"].item() / count),
        "n_img": float(stats["n_img"].item() / count),
        "c_vis": float(stats["c_vis"].item() / count),
        "c_text": float(stats["c_text"].item() / count),
    }


def _analyze_dataset(
    r3: R3,
    dataset: Dataset,
    cfg: TrainConfig,
    levels: List[float],
    use_pseudo_text: bool,
    rank: int,
    world_size: int,
    is_main_process: bool,
) -> Dict[str, Any]:
    dataset = _shard_dataset(dataset, rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        collate_fn=UnifiedQACollator(
            tokenizer=None,
            max_length=cfg.data.max_length,
            image_root=cfg.data.image_root,
            image_size=cfg.data.image_size,
        ),
    )
    results: Dict[str, Any] = {}
    for level in levels:
        stats = {
            "count": torch.tensor(0.0, device=r3.qwen.device),
            "gate_sum": torch.zeros(3, device=r3.qwen.device),
            "gate_sumsq": torch.zeros(3, device=r3.qwen.device),
            "gate_entropy": torch.tensor(0.0, device=r3.qwen.device),
            "retrieval_mass": torch.tensor(0.0, device=r3.qwen.device),
            "latent_mass": torch.tensor(0.0, device=r3.qwen.device),
            "n_text": torch.tensor(0.0, device=r3.qwen.device),
            "n_img": torch.tensor(0.0, device=r3.qwen.device),
            "c_vis": torch.tensor(0.0, device=r3.qwen.device),
            "c_text": torch.tensor(0.0, device=r3.qwen.device),
        }
        for batch in loader:
            images = batch["corrupted"]["images"]
            questions = batch["corrupted"]["questions"]
            pseudo_texts = batch["corrupted"]["pseudo_texts"]
            with torch.no_grad():
                gates, c_vis, c_text = _compute_gates_only(
                    r3,
                    images,
                    questions,
                    pseudo_texts,
                    corruption_level=level,
                    top_k=cfg.retrieval.top_k,
                    use_pseudo_text=use_pseudo_text,
                )
            gates = gates.float()
            count = gates.size(0)
            stats["count"] += float(count)
            stats["gate_sum"] += gates.sum(dim=0)
            stats["gate_sumsq"] += (gates ** 2).sum(dim=0)
            entropy = -(gates * torch.log(gates.clamp_min(1e-8))).sum(dim=-1)
            stats["gate_entropy"] += entropy.sum()
            stats["retrieval_mass"] += (gates[:, 0] + gates[:, 1]).sum()
            stats["latent_mass"] += gates[:, 2].sum()
            n_text = torch.clamp(torch.round(gates[:, 0] * cfg.retrieval.top_k), min=1.0)
            n_img = torch.clamp(torch.round(gates[:, 1] * cfg.retrieval.top_k), min=0.0)
            stats["n_text"] += n_text.sum()
            stats["n_img"] += n_img.sum()
            stats["c_vis"] += c_vis.view(-1).sum()
            stats["c_text"] += c_text.view(-1).sum()

        stats = _reduce_stats(stats)
        results[f"{level:.2f}"] = _finalize_stats(stats)
        if is_main_process:
            summary = results[f"{level:.2f}"]
            logging.info(
                "Gate stats level %.2f | g_mean=%s | retrieval=%.3f latent=%.3f",
                level,
                ",".join(f"{v:.3f}" for v in summary["gate_mean"]),
                summary["retrieval_mass"],
                summary["latent_mass"],
            )
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Analyze R3 gate weights")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--index_dir", default="indices")
    parser.add_argument("--out_json", default="results/gate_stats.json")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset_prefixes", default=None)
    parser.add_argument("--corruption_levels", default="0.0,0.2,0.4,0.8,1.0")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--use_pseudo_text", action="store_true")
    parser.add_argument("--no_pseudo_text", action="store_true")
    parser.add_argument("--disable_corruption", action="store_true")
    parser.add_argument("--corruption_max_severity", type=float, default=None)
    parser.add_argument("--blur_prob", type=float, default=None)
    parser.add_argument("--motion_blur_prob", type=float, default=None)
    parser.add_argument("--occlusion_prob", type=float, default=None)
    parser.add_argument("--crop_prob", type=float, default=None)
    parser.add_argument("--downsample_prob", type=float, default=None)
    parser.add_argument("--jpeg_prob", type=float, default=None)
    parser.add_argument("--noise_prob", type=float, default=None)
    parser.add_argument("--color_prob", type=float, default=None)
    parser.add_argument("--text_trunc_prob", type=float, default=None)
    parser.add_argument("--text_noise_prob", type=float, default=None)
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument("--jpeg_quality_min", type=int, default=None)
    parser.add_argument("--jpeg_quality_max", type=int, default=None)
    parser.add_argument("--color_jitter", type=float, default=None)
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
    if args.max_length is not None:
        cfg.data.max_length = args.max_length
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.max_eval_samples is not None:
        cfg.evaluation.max_eval_samples = args.max_eval_samples
    if args.batch_size is not None:
        cfg.evaluation.batch_size = args.batch_size
    cfg.retrieval.index_dir = args.index_dir
    cfg.retrieval.image_index_path = f"{args.index_dir}/image.index"
    cfg.retrieval.image_meta_path = f"{args.index_dir}/image.meta.json"
    cfg.retrieval.image_embeds_path = f"{args.index_dir}/image.embeds.npy"
    cfg.retrieval.text_index_path = f"{args.index_dir}/text.index"
    cfg.retrieval.text_meta_path = f"{args.index_dir}/text.meta.json"
    cfg.retrieval.text_embeds_path = f"{args.index_dir}/text.embeds.npy"
    cfg.retrieval.top_k = args.top_k
    cfg.r3.enable_corruption = not args.disable_corruption
    if args.corruption_max_severity is not None:
        cfg.r3.corruption.max_severity = args.corruption_max_severity
    if args.blur_prob is not None:
        cfg.r3.corruption.blur_prob = args.blur_prob
    if args.motion_blur_prob is not None:
        cfg.r3.corruption.motion_blur_prob = args.motion_blur_prob
    if args.occlusion_prob is not None:
        cfg.r3.corruption.occlusion_prob = args.occlusion_prob
    if args.crop_prob is not None:
        cfg.r3.corruption.crop_prob = args.crop_prob
    if args.downsample_prob is not None:
        cfg.r3.corruption.downsample_prob = args.downsample_prob
    if args.jpeg_prob is not None:
        cfg.r3.corruption.jpeg_prob = args.jpeg_prob
    if args.noise_prob is not None:
        cfg.r3.corruption.noise_prob = args.noise_prob
    if args.color_prob is not None:
        cfg.r3.corruption.color_prob = args.color_prob
    if args.text_trunc_prob is not None:
        cfg.r3.corruption.text_trunc_prob = args.text_trunc_prob
    if args.text_noise_prob is not None:
        cfg.r3.corruption.text_noise_prob = args.text_noise_prob
    if args.noise_std is not None:
        cfg.r3.corruption.noise_std = args.noise_std
    if args.jpeg_quality_min is not None:
        cfg.r3.corruption.jpeg_quality_min = args.jpeg_quality_min
    if args.jpeg_quality_max is not None:
        cfg.r3.corruption.jpeg_quality_max = args.jpeg_quality_max
    if args.color_jitter is not None:
        cfg.r3.corruption.color_jitter = args.color_jitter

    if distributed:
        cfg.model.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        cfg.model.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        cfg.model.use_lora = True
        qwen.load_lora_adapter(args.checkpoint_dir)

    text_retriever = None
    image_retriever = None
    text_index_ready = os.path.exists(cfg.retrieval.text_index_path) and os.path.exists(
        cfg.retrieval.text_meta_path
    )
    image_index_ready = os.path.exists(cfg.retrieval.image_index_path) and os.path.exists(
        cfg.retrieval.image_meta_path
    )
    if text_index_ready:
        text_retriever = TextRetriever(cfg.retrieval.text_encoder_name)
        text_retriever.load(
            cfg.retrieval.text_index_path,
            cfg.retrieval.text_meta_path,
            cfg.retrieval.text_embeds_path,
        )
    if image_index_ready:
        image_retriever = ImageRetriever(
            cfg.retrieval.image_encoder_name, device=cfg.model.device
        )
        image_retriever.load(
            cfg.retrieval.image_index_path,
            cfg.retrieval.image_meta_path,
            cfg.retrieval.image_embeds_path,
        )

    r3 = R3(qwen, text_retriever, image_retriever, cfg.r3)
    r3_state = os.path.join(args.checkpoint_dir, "r3_state.pt")
    if os.path.exists(r3_state):
        r3.load_state_dict(torch.load(r3_state, map_location=cfg.model.device), strict=False)
    r3.to(cfg.model.device)
    r3.eval()

    levels = _parse_levels(args.corruption_levels)
    if args.use_pseudo_text and args.no_pseudo_text:
        raise ValueError("Cannot set both --use_pseudo_text and --no_pseudo_text")
    use_pseudo_text = not args.no_pseudo_text
    if args.use_pseudo_text:
        use_pseudo_text = True

    if args.dataset and args.dataset_prefixes:
        raise ValueError("Set either --dataset or --dataset_prefixes, not both.")

    results: Dict[str, Any] = {}
    if args.dataset or args.dataset_prefixes:
        prefixes = [args.dataset] if args.dataset else [
            p.strip() for p in args.dataset_prefixes.split(",") if p.strip()
        ]
        grouped = _load_grouped_samples(
            cfg.data.val_jsonl, prefixes, cfg.evaluation.max_eval_samples
        )
        for prefix in prefixes:
            samples = grouped.get(prefix, [])
            if is_main_process:
                logging.info("Analyze dataset=%s | samples=%d", prefix, len(samples))
            dataset = _ListQADataset(samples)
            results[prefix] = _analyze_dataset(
                r3, dataset, cfg, levels, use_pseudo_text, rank, world_size, is_main_process
            )
    else:
        dataset = UnifiedQADataset(
            cfg.data.val_jsonl,
            image_root=cfg.data.image_root,
            max_samples=cfg.evaluation.max_eval_samples,
        )
        results["all"] = _analyze_dataset(
            r3, dataset, cfg, levels, use_pseudo_text, rank, world_size, is_main_process
        )

    if is_main_process:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
    if distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
