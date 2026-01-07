"""Compute proxy evidence-supported answer rates (ESAR)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from models.r3_modules import CorruptionSimulator, R3
from retrieval.image_retrieval import ImageRetriever
from retrieval.text_retrieval import TextRetriever


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


class _ListQADataset(Dataset):
    def __init__(self, samples: List[UnifiedQADatum]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> UnifiedQADatum:
        return self.samples[idx]


def _load_ocr_map(ocr_jsonl: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(ocr_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            image_path = data.get("image_path")
            if not image_path or image_path in mapping:
                continue
            text = data.get("ocr_text") or data.get("text") or ""
            mapping[image_path] = str(text)
    return mapping


def _numeric_match(pred: str, evidence: str, num_tol: float) -> bool:
    pred_num = eval_utils._try_parse_float(pred)
    if pred_num is None:
        return False
    evidence_norm = eval_utils._canonical_text(evidence)
    for match in re.findall(r"-?\\d+(?:\\.\\d+)?", evidence_norm):
        try:
            ev_num = float(match)
        except ValueError:
            continue
        denom = max(abs(pred_num), 1e-6)
        if abs(ev_num - pred_num) / denom <= num_tol:
            return True
    return False


def _tokens_subset(pred_tokens: List[str], evidence_tokens: List[str]) -> bool:
    if not pred_tokens or not evidence_tokens:
        return False
    pred_counts = Counter(pred_tokens)
    ev_counts = Counter(evidence_tokens)
    return all(ev_counts[tok] >= pred_counts[tok] for tok in pred_counts)


def _support_flags(pred: str, evidence: str, num_tol: float) -> Tuple[bool, bool]:
    pred = pred.strip()
    evidence = evidence.strip()
    if not pred or not evidence:
        return False, False
    pred_norm = eval_utils._canonical_text(pred)
    ev_norm = eval_utils._canonical_text(evidence)
    if not pred_norm or not ev_norm:
        return False, False
    strict = pred_norm in ev_norm
    if strict:
        return True, True
    if _numeric_match(pred, evidence, num_tol=num_tol):
        return False, True
    pred_tokens = eval_utils._canonical_tokens(pred)
    ev_tokens = eval_utils._canonical_tokens(evidence)
    relaxed = _tokens_subset(pred_tokens, ev_tokens)
    return False, relaxed


def _supported(pred: str, evidence: str, num_tol: float) -> Tuple[bool, bool]:
    candidates = eval_utils._candidate_spans(pred)
    if not candidates:
        candidates = [pred]
    strict_any = False
    relaxed_any = False
    for cand in candidates:
        strict, relaxed = _support_flags(cand, evidence, num_tol=num_tol)
        strict_any = strict_any or strict
        relaxed_any = relaxed_any or relaxed
        if strict_any and relaxed_any:
            break
    return strict_any, relaxed_any


def _reduce_counts(
    counts: Dict[float, Dict[str, float]],
    device: str,
) -> Dict[float, Dict[str, float]]:
    keys = [
        "count",
        "ocr_strict",
        "ocr_relaxed",
        "ret_strict",
        "ret_relaxed",
        "any_strict",
        "any_relaxed",
    ]
    reduced: Dict[float, Dict[str, float]] = {}
    for level, metrics in counts.items():
        values = [metrics.get(k, 0.0) for k in keys]
        tensor = torch.tensor(values, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced[level] = {keys[i]: tensor[i].item() for i in range(len(keys))}
    return reduced


def _summarize(counts: Dict[float, Dict[str, float]]) -> Dict[float, Dict[str, float]]:
    out: Dict[float, Dict[str, float]] = {}
    for level, metrics in counts.items():
        total = max(metrics.get("count", 0.0), 1.0)
        out[level] = {
            "count": metrics.get("count", 0.0),
            "ocr_strict": metrics.get("ocr_strict", 0.0) / total,
            "ocr_relaxed": metrics.get("ocr_relaxed", 0.0) / total,
            "ret_strict": metrics.get("ret_strict", 0.0) / total,
            "ret_relaxed": metrics.get("ret_relaxed", 0.0) / total,
            "any_strict": metrics.get("any_strict", 0.0) / total,
            "any_relaxed": metrics.get("any_relaxed", 0.0) / total,
        }
    return out


def _update_dataclass(cfg: Any, updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if hasattr(cfg, key):
            sub = getattr(cfg, key)
            if isinstance(value, dict) and hasattr(sub, "__dict__"):
                _update_dataclass(sub, value)
            else:
                setattr(cfg, key, value)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Compute ESAR (evidence-supported answer rate)")
    parser.add_argument("--eval_mode", choices=["r3", "base"], default="r3")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--index_dir", default="indices")
    parser.add_argument("--ocr_jsonl", default=None)
    parser.add_argument("--out_json", default="esar_results.json")
    parser.add_argument("--out_jsonl", default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--corruption_levels", default="0.0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--disable_corruption", action="store_true")
    parser.add_argument(
        "--corrupt_text_target",
        choices=["pseudo_text", "question", "none"],
        default="pseudo_text",
    )
    parser.add_argument("--answer_only", action="store_true")
    parser.add_argument("--use_pseudo_text", action="store_true")
    parser.add_argument("--no_pseudo_text", action="store_true")
    parser.add_argument("--dataset_prefixes", default=None)
    parser.add_argument("--num_tol", type=float, default=0.01)
    args = parser.parse_args()

    distributed = _is_distributed()
    rank = 0
    world_size = 1
    local_rank = 0
    if distributed:
        rank, world_size, local_rank = _init_distributed()
    is_main_process = rank == 0

    if args.use_pseudo_text and args.no_pseudo_text:
        raise ValueError("Cannot set both --use_pseudo_text and --no_pseudo_text")
    use_pseudo_text: Optional[bool] = None
    if args.use_pseudo_text:
        use_pseudo_text = True
    if args.no_pseudo_text:
        use_pseudo_text = False
    if use_pseudo_text is None:
        use_pseudo_text = args.eval_mode == "r3"

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

    if distributed:
        cfg.model.device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if args.model_name:
        cfg.model.model_name = args.model_name
    elif args.checkpoint_dir and not cfg.model.model_name:
        cfg.model.model_name = args.checkpoint_dir

    adapter_path = None
    if args.checkpoint_dir:
        adapter_safetensors = os.path.join(args.checkpoint_dir, "adapter_model.safetensors")
        adapter_bin = os.path.join(args.checkpoint_dir, "adapter_model.bin")
        if os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin):
            adapter_path = args.checkpoint_dir
            if args.eval_mode == "r3":
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
        elif cfg.r3.enable_image_retrieval:
            raise FileNotFoundError(
                f"Missing image index files under {cfg.retrieval.index_dir}"
            )
        r3 = R3(qwen, text_retriever, image_retriever, cfg.r3)
        if args.checkpoint_dir:
            r3_state = os.path.join(args.checkpoint_dir, "r3_state.pt")
            if os.path.exists(r3_state):
                r3.load_state_dict(torch.load(r3_state, map_location=qwen.device), strict=False)
        r3.to(qwen.device)

    ocr_map = _load_ocr_map(args.ocr_jsonl) if args.ocr_jsonl else {}

    if args.dataset_prefixes:
        prefixes = [p.strip() for p in args.dataset_prefixes.split(",") if p.strip()]
        grouped = _load_grouped_samples(args.val_jsonl, prefixes, cfg.evaluation.max_eval_samples)
    else:
        prefixes = ["all"]
        grouped = {
            "all": UnifiedQADataset(
                cfg.data.val_jsonl,
                image_root=cfg.data.image_root,
                max_samples=cfg.evaluation.max_eval_samples,
            ).samples
        }

    results: Dict[str, Dict[float, Dict[str, float]]] = {}
    levels = _parse_levels(args.corruption_levels)
    for prefix in prefixes:
        samples = grouped.get(prefix, [])
        if prefix == "all":
            dataset = UnifiedQADataset(
                args.val_jsonl,
                image_root=args.image_root,
                max_samples=cfg.evaluation.max_eval_samples,
            )
        else:
            dataset = _ListQADataset(samples)
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

        if args.eval_mode == "base":
            corruptor = CorruptionSimulator(cfg.r3)
        else:
            corruptor = None
        counts: Dict[float, Dict[str, float]] = {lvl: Counter() for lvl in levels}
        out_f = None
        if args.out_jsonl:
            out_path = args.out_jsonl
            if world_size > 1:
                out_path = f"{args.out_jsonl}.rank{rank}"
            out_f = open(out_path, "w", encoding="utf-8")

        for level in levels:
            if is_main_process:
                logging.info("ESAR dataset=%s level=%.2f", prefix, level)
            for batch in dataloader:
                clean = batch["clean"]
                corrupted = batch["corrupted"]
                images = corrupted["images"]
                questions = corrupted["questions"]
                pseudo_texts = corrupted["pseudo_texts"]
                image_paths = clean["image_paths"]

                if args.eval_mode == "base":
                    images, questions, pseudo_texts = eval_utils._apply_corruption(
                        corruptor, images, questions, pseudo_texts, level, args.corrupt_text_target
                    )
                    with torch.no_grad():
                        preds = qwen.generate_answer(
                            images,
                            questions,
                            pseudo_texts if use_pseudo_text else None,
                            max_new_tokens=cfg.evaluation.max_new_tokens,
                            answer_only=args.answer_only,
                        )
                    retrieved_texts = [[] for _ in preds]
                    contexts = [""] * len(preds)
                else:
                    use_pseudo = pseudo_texts if use_pseudo_text else ["" for _ in pseudo_texts]
                    with torch.no_grad():
                        preds, retrieved_texts, _, contexts, _ = r3.generate(
                            images,
                            questions,
                            use_pseudo,
                            corruption_level=level,
                            top_k=cfg.retrieval.top_k,
                            max_new_tokens=cfg.evaluation.max_new_tokens,
                            return_retrieval=True,
                            answer_only=args.answer_only,
                        )

                for pred, pseudo, question, image_path, ctx, rtexts in zip(
                    preds, pseudo_texts, questions, image_paths, contexts, retrieved_texts
                ):
                    ocr_text = ocr_map.get(image_path, pseudo)
                    retrieval_text = " ".join(rtexts).strip()
                    if ctx:
                        retrieval_text = f"{retrieval_text} {ctx}".strip()
                    strict_ocr, relaxed_ocr = _supported(pred, ocr_text, args.num_tol)
                    strict_ret, relaxed_ret = _supported(pred, retrieval_text, args.num_tol)
                    strict_any = strict_ocr or strict_ret
                    relaxed_any = relaxed_ocr or relaxed_ret
                    counts[level].update(
                        {
                            "count": 1.0,
                            "ocr_strict": float(strict_ocr),
                            "ocr_relaxed": float(relaxed_ocr),
                            "ret_strict": float(strict_ret),
                            "ret_relaxed": float(relaxed_ret),
                            "any_strict": float(strict_any),
                            "any_relaxed": float(relaxed_any),
                        }
                    )
                    if out_f is not None and is_main_process:
                        out_f.write(
                            json.dumps(
                                {
                                        "image_path": image_path,
                                        "question": question,
                                        "pred": pred,
                                        "ocr_text": ocr_text,
                                        "retrieval_text": retrieval_text,
                                    "support_strict": strict_any,
                                    "support_relaxed": relaxed_any,
                                }
                            )
                            + "\n"
                        )
        if out_f is not None:
            out_f.close()

        if distributed:
            counts = _reduce_counts(counts, device=cfg.model.device)
        results[prefix] = _summarize(counts)

    if is_main_process:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
    if distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
