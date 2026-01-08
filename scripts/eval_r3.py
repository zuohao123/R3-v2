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
import torch.distributed as dist
from data.datasets import UnifiedQACollator, UnifiedQADataset, UnifiedQADatum
from torch.utils.data import DataLoader, Dataset, Subset


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


def _reduce_metrics(
    results: Dict[float, Dict[str, float]], device: str
) -> Dict[float, Dict[str, float]]:
    metric_keys = ["exact_match", "f1", "bleu", "rouge_l", "anls", "relaxed_acc"]
    reduced: Dict[float, Dict[str, float]] = {}
    for level, metrics in results.items():
        values = [metrics.get(k, 0.0) for k in metric_keys]
        values.append(metrics.get("count", 0.0))
        tensor = torch.tensor(values, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        count = tensor[-1].item()
        denom = max(count, 1.0)
        reduced[level] = {
            metric_keys[i]: (tensor[i].item() / denom) for i in range(len(metric_keys))
        }
    return reduced


def _log_results(tag: str, results: Dict[float, Dict[str, float]]) -> None:
    logger = logging.getLogger(__name__)
    for level in sorted(results.keys()):
        metrics = results[level]
        logger.info(
            "Eval summary %s | corruption=%.2f | EM %.4f F1 %.4f BLEU %.4f ROUGE-L %.4f ANLS %.4f RA %.4f",
            tag,
            level,
            metrics.get("exact_match", 0.0),
            metrics.get("f1", 0.0),
            metrics.get("bleu", 0.0),
            metrics.get("rouge_l", 0.0),
            metrics.get("anls", 0.0),
            metrics.get("relaxed_acc", 0.0),
        )

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
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--min_text_score", type=float, default=None)
    parser.add_argument("--min_image_score", type=float, default=None)
    parser.add_argument("--max_text_score", type=float, default=None)
    parser.add_argument("--max_image_score", type=float, default=None)
    parser.add_argument("--shuffle_retrieval", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=None)
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
    parser.add_argument("--sample_every", type=int, default=None)
    parser.add_argument("--sample_max", type=int, default=None)
    parser.add_argument("--answer_only", action="store_true")
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
        "--corruption_max_severity",
        type=float,
        default=None,
        help="Scale corruption intensity (effective_level = level * max_severity).",
    )
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
    parser.add_argument("--enable_router", action="store_true")
    parser.add_argument("--disable_router", action="store_true")
    parser.add_argument("--router_hidden", type=int, default=None)
    parser.add_argument("--router_dropout", type=float, default=None)
    parser.add_argument("--router_out_dim", type=int, default=None)
    parser.add_argument(
        "--load_lora_adapter",
        action="store_true",
        help="Load LoRA adapter weights from --checkpoint_dir if present.",
    )
    parser.add_argument("--max_context_chars", type=int, default=None)
    args = parser.parse_args()

    distributed = _is_distributed()
    rank = 0
    world_size = 1
    local_rank = 0
    if distributed:
        rank, world_size, local_rank = _init_distributed()
    is_main_process = rank == 0

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
    if args.max_new_tokens is not None:
        cfg.evaluation.max_new_tokens = args.max_new_tokens
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
    cfg.r3.enable_text_retrieval = not args.disable_text_retrieval
    cfg.r3.enable_image_retrieval = not args.disable_image_retrieval
    cfg.r3.enable_prefix = not args.disable_prefix
    cfg.r3.enable_memory = not args.disable_memory
    cfg.r3.enable_visual_memory = not args.disable_visual_memory
    cfg.r3.enable_latent_tokens = not args.disable_latent_tokens
    cfg.r3.enable_gate = not args.disable_gate
    cfg.r3.enable_context = not args.disable_context
    if args.enable_router:
        cfg.r3.enable_router = True
    if args.disable_router:
        cfg.r3.enable_router = False
    if args.router_hidden is not None:
        cfg.r3.router_hidden = args.router_hidden
    if args.router_dropout is not None:
        cfg.r3.router_dropout = args.router_dropout
    if args.router_out_dim is not None:
        cfg.r3.router_out_dim = args.router_out_dim
    if args.min_text_score is not None:
        cfg.r3.min_text_score = args.min_text_score
    if args.min_image_score is not None:
        cfg.r3.min_image_score = args.min_image_score
    if args.max_text_score is not None:
        cfg.r3.max_text_score = args.max_text_score
    if args.max_image_score is not None:
        cfg.r3.max_image_score = args.max_image_score
    if args.shuffle_retrieval:
        cfg.r3.shuffle_retrieval = True
    if args.shuffle_seed is not None:
        cfg.r3.shuffle_seed = args.shuffle_seed
    if args.max_context_chars is not None:
        cfg.r3.max_context_chars = args.max_context_chars
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
    if args.use_soft_prefix:
        cfg.r3.use_soft_prefix = True
    if args.disable_soft_prefix:
        cfg.r3.use_soft_prefix = False

    if distributed:
        cfg.model.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
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
            if not cfg.r3.enable_text_retrieval and is_main_process:
                logging.info(
                    "Text retrieval disabled; loaded text index to match R3 dimensions."
                )
        elif cfg.r3.enable_text_retrieval:
            raise FileNotFoundError(
                f"Missing text index files under {cfg.retrieval.index_dir}"
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
            if not cfg.r3.enable_image_retrieval and is_main_process:
                logging.info(
                    "Image retrieval disabled; loaded image index to match R3 dimensions."
                )
        elif cfg.r3.enable_image_retrieval:
            raise FileNotFoundError(
                f"Missing image index files under {cfg.retrieval.index_dir}"
            )

        r3 = R3(qwen, text_retriever, image_retriever, cfg.r3)
        if args.checkpoint_dir:
            r3_state = os.path.join(args.checkpoint_dir, "r3_state.pt")
            if os.path.exists(r3_state):
                r3.load_state_dict(
                    torch.load(r3_state, map_location=qwen.device),
                    strict=False,
                )
        r3.to(qwen.device)

    dataset = UnifiedQADataset(
        cfg.data.val_jsonl,
        image_root=cfg.data.image_root,
        max_samples=cfg.evaluation.max_eval_samples,
    )
    dataset = _shard_dataset(dataset, rank, world_size)
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
            if is_main_process:
                logging.info("Eval dataset=%s | samples=%d", prefix, len(samples))
            subset = _ListQADataset(samples)
            subset = _shard_dataset(subset, rank, world_size)
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
            results = evaluate_model(
                model,
                subset_loader,
                corruption_levels=levels,
                max_new_tokens=cfg.evaluation.max_new_tokens,
                top_k=cfg.retrieval.top_k,
                return_sums=distributed,
                mode=args.eval_mode,
                use_pseudo_text=use_pseudo_text,
                corrupt_text_target=args.corrupt_text_target,
                corruptor=corruptor,
                log_every=args.eval_log_every,
                sample_every=args.sample_every,
                sample_max=args.sample_max,
                is_main_process=is_main_process,
                answer_only=args.answer_only,
            )
            if distributed:
                results = _reduce_metrics(results, device=cfg.model.device)
            if is_main_process:
                _log_results(f"dataset={prefix}", results)
                results_by_dataset[prefix] = results
        if is_main_process:
            save_results(results_by_dataset, args.out_json)
            print(json.dumps(results_by_dataset, indent=2))
        if distributed:
            dist.barrier()
        return

    results = evaluate_model(
        model,
        dataloader,
        corruption_levels=levels,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        top_k=cfg.retrieval.top_k,
        return_sums=distributed,
        mode=args.eval_mode,
        use_pseudo_text=use_pseudo_text,
        corrupt_text_target=args.corrupt_text_target,
        corruptor=corruptor,
        log_every=args.eval_log_every,
        sample_every=args.sample_every,
        sample_max=args.sample_max,
        is_main_process=is_main_process,
        answer_only=args.answer_only,
    )
    if distributed:
        results = _reduce_metrics(results, device=cfg.model.device)
    if is_main_process:
        _log_results("all", results)
        save_results(results, args.out_json)
        print(json.dumps(results, indent=2))
    if distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
