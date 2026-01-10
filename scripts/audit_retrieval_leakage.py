#!/usr/bin/env python
"""Audit retrieval leakage: answer-hit, near-duplicate, and overlap rates."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image

from config.train_config import TrainConfig
from data.datasets import UnifiedQADatum
from evaluation.evaluate import _canonical_tokens
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


def _load_samples(jsonl_path: str, max_samples: Optional[int]) -> List[UnifiedQADatum]:
    samples: List[UnifiedQADatum] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            samples.append(
                UnifiedQADatum(
                    image_path=data["image_path"],
                    question=data["question"],
                    answer=data["answer"],
                    pseudo_text=data.get("pseudo_text", ""),
                )
            )
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def _resolve_path(image_root: str, path: str) -> str:
    if image_root and not os.path.isabs(path):
        return os.path.join(image_root, path)
    return path


def _normalize_tokens(text: str) -> List[str]:
    return _canonical_tokens(text or "")


def _answers_list(answer: Any) -> List[str]:
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(a) for a in answer if str(a).strip()]
    return [str(answer)] if str(answer).strip() else []


def _contains_answer(text: str, answers: List[str]) -> bool:
    if not answers:
        return False
    text_tokens = _normalize_tokens(text)
    if not text_tokens:
        return False
    text_str = " ".join(text_tokens)
    padded = f" {text_str} "
    for ans in answers:
        ans_tokens = _normalize_tokens(ans)
        if not ans_tokens:
            continue
        ans_str = " ".join(ans_tokens)
        if f" {ans_str} " in padded:
            return True
    return False


def _token_overlap_ratio(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    qset = set(query_tokens)
    dset = set(doc_tokens)
    if not qset:
        return 0.0
    return len(qset & dset) / max(len(qset), 1)


def _dhash(image: Image.Image, size: int = 8) -> int:
    # Difference hash (dHash) for lightweight layout similarity.
    img = image.convert("L").resize((size + 1, size), Image.BILINEAR)
    pixels = np.asarray(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def _hash_similarity(a: int, b: int, nbits: int = 64) -> float:
    return 1.0 - ((a ^ b).bit_count() / nbits)


def _iter_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _build_query(
    question: str, pseudo_text: str, mode: str
) -> str:
    if mode == "question":
        return question
    if mode == "pseudo_text":
        return pseudo_text
    combined = f"{question} {pseudo_text}".strip()
    return combined if combined else question


def _audit_text(
    samples: List[UnifiedQADatum],
    retriever: TextRetriever,
    top_k: int,
    query_mode: str,
    batch_size: int,
    sim_thresholds: List[float],
    overlap_thresholds: List[float],
) -> Dict[str, float]:
    hit_k = 0
    hit_1 = 0
    count = 0
    top1_scores: List[float] = []
    max_scores: List[float] = []
    near_dup = {thr: 0 for thr in sim_thresholds}
    overlap_top1: List[float] = []
    overlap_max: List[float] = []
    overlap_near_dup = {thr: 0 for thr in overlap_thresholds}
    for batch in _iter_batches(samples, batch_size):
        queries = [
            _build_query(s.question, s.pseudo_text, query_mode) for s in batch
        ]
        result = retriever.retrieve(queries, top_k)
        metas = result["metadata"]
        scores = result.get("scores")
        for sample, retrieved in zip(batch, metas):
            answers = _answers_list(sample.answer)
            texts = [
                (meta.get("full_pseudo_text") or meta.get("pseudo_text") or meta.get("ocr_text") or "")
                for meta in retrieved
            ]
            query_tokens = _normalize_tokens(sample.pseudo_text or "")
            overlap_scores = [
                _token_overlap_ratio(query_tokens, _normalize_tokens(t)) for t in texts
            ]
            count += 1
            if texts and _contains_answer(texts[0], answers):
                hit_1 += 1
            if any(_contains_answer(t, answers) for t in texts):
                hit_k += 1
            if overlap_scores:
                overlap_top1.append(overlap_scores[0])
                max_overlap = max(overlap_scores)
                overlap_max.append(max_overlap)
                for thr in overlap_thresholds:
                    if max_overlap >= thr:
                        overlap_near_dup[thr] += 1
            else:
                overlap_top1.append(0.0)
                overlap_max.append(0.0)
            if scores is not None:
                row = scores[min(len(top1_scores), len(scores) - 1)]
                if row.size > 0:
                    top1 = float(row[0])
                    mscore = float(row.max())
                    top1_scores.append(top1)
                    max_scores.append(mscore)
                    for thr in sim_thresholds:
                        if mscore >= thr:
                            near_dup[thr] += 1
                else:
                    top1_scores.append(0.0)
                    max_scores.append(0.0)
    return {
        "count": count,
        "answer_hit_at_1": hit_1 / max(count, 1),
        "answer_hit_at_k": hit_k / max(count, 1),
        "top1_sim_mean": float(np.mean(top1_scores)) if top1_scores else 0.0,
        "top1_sim_p95": float(np.percentile(top1_scores, 95)) if top1_scores else 0.0,
        "max_sim_mean": float(np.mean(max_scores)) if max_scores else 0.0,
        "max_sim_p95": float(np.percentile(max_scores, 95)) if max_scores else 0.0,
        "near_dup_rate": {str(thr): near_dup[thr] / max(count, 1) for thr in sim_thresholds},
        "ocr_overlap_top1_mean": float(np.mean(overlap_top1)) if overlap_top1 else 0.0,
        "ocr_overlap_top1_p95": float(np.percentile(overlap_top1, 95)) if overlap_top1 else 0.0,
        "ocr_overlap_max_mean": float(np.mean(overlap_max)) if overlap_max else 0.0,
        "ocr_overlap_max_p95": float(np.percentile(overlap_max, 95)) if overlap_max else 0.0,
        "ocr_overlap_near_dup_rate": {
            str(thr): overlap_near_dup[thr] / max(count, 1) for thr in overlap_thresholds
        },
    }


def _audit_image(
    samples: List[UnifiedQADatum],
    retriever: ImageRetriever,
    top_k: int,
    image_root: str,
    batch_size: int,
    sim_thresholds: List[float],
    hash_thresholds: List[float],
) -> Dict[str, float]:
    hit_k = 0
    hit_1 = 0
    count = 0
    top1_scores: List[float] = []
    max_scores: List[float] = []
    near_dup = {thr: 0 for thr in sim_thresholds}
    hash_top1: List[float] = []
    hash_max: List[float] = []
    hash_near_dup = {thr: 0 for thr in hash_thresholds}
    for batch in _iter_batches(samples, batch_size):
        images: List[Image.Image] = []
        paths: List[str] = []
        hashes: List[int] = []
        for sample in batch:
            path = _resolve_path(image_root, sample.image_path)
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                hashes.append(_dhash(img))
                paths.append(sample.image_path)
            except (FileNotFoundError, OSError):
                continue
        if not images:
            continue
        result = retriever.retrieve(images, top_k)
        metas = result["metadata"]
        scores = result.get("scores")
        for path, qhash, retrieved in zip(paths, hashes, metas):
            count += 1
            retrieved_paths = [
                (meta.get("image_path") or "").replace("\\", "/").lstrip("./")
                for meta in retrieved
            ]
            target = path.replace("\\", "/").lstrip("./")
            if retrieved_paths and retrieved_paths[0] == target:
                hit_1 += 1
            if target in retrieved_paths:
                hit_k += 1
            retrieved_hash_scores: List[float] = []
            for meta in retrieved:
                rpath = meta.get("image_path")
                if not rpath:
                    retrieved_hash_scores.append(0.0)
                    continue
                full_path = _resolve_path(image_root, rpath)
                try:
                    rimg = Image.open(full_path).convert("RGB")
                    rhash = _dhash(rimg)
                    retrieved_hash_scores.append(_hash_similarity(qhash, rhash))
                except (FileNotFoundError, OSError):
                    retrieved_hash_scores.append(0.0)
            if retrieved_hash_scores:
                hash_top1.append(retrieved_hash_scores[0])
                max_hash = max(retrieved_hash_scores)
                hash_max.append(max_hash)
                for thr in hash_thresholds:
                    if max_hash >= thr:
                        hash_near_dup[thr] += 1
            else:
                hash_top1.append(0.0)
                hash_max.append(0.0)
            if scores is not None:
                row = scores[min(len(top1_scores), len(scores) - 1)]
                if row.size > 0:
                    top1 = float(row[0])
                    mscore = float(row.max())
                    top1_scores.append(top1)
                    max_scores.append(mscore)
                    for thr in sim_thresholds:
                        if mscore >= thr:
                            near_dup[thr] += 1
                else:
                    top1_scores.append(0.0)
                    max_scores.append(0.0)
    return {
        "count": count,
        "self_hit_at_1": hit_1 / max(count, 1),
        "self_hit_at_k": hit_k / max(count, 1),
        "top1_sim_mean": float(np.mean(top1_scores)) if top1_scores else 0.0,
        "top1_sim_p95": float(np.percentile(top1_scores, 95)) if top1_scores else 0.0,
        "max_sim_mean": float(np.mean(max_scores)) if max_scores else 0.0,
        "max_sim_p95": float(np.percentile(max_scores, 95)) if max_scores else 0.0,
        "near_dup_rate": {str(thr): near_dup[thr] / max(count, 1) for thr in sim_thresholds},
        "hash_sim_top1_mean": float(np.mean(hash_top1)) if hash_top1 else 0.0,
        "hash_sim_top1_p95": float(np.percentile(hash_top1, 95)) if hash_top1 else 0.0,
        "hash_sim_max_mean": float(np.mean(hash_max)) if hash_max else 0.0,
        "hash_sim_max_p95": float(np.percentile(hash_max, 95)) if hash_max else 0.0,
        "hash_sim_near_dup_rate": {
            str(thr): hash_near_dup[thr] / max(count, 1) for thr in hash_thresholds
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Audit retrieval leakage")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--index_dir", default="indices")
    parser.add_argument("--out_json", default="results/retrieval_audit.json")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset_prefixes", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--query_mode", choices=["concat", "question", "pseudo_text"], default="concat")
    parser.add_argument(
        "--sim_thresholds",
        default="0.9,0.8",
        help="Comma-separated similarity thresholds for near-duplicate rate.",
    )
    parser.add_argument(
        "--text_overlap_thresholds",
        default="0.5,0.7",
        help="Comma-separated OCR overlap thresholds for near-duplicate rate.",
    )
    parser.add_argument(
        "--hash_sim_thresholds",
        default="0.9,0.8",
        help="Comma-separated dHash similarity thresholds for near-duplicate rate.",
    )
    parser.add_argument("--skip_text", action="store_true")
    parser.add_argument("--skip_image", action="store_true")
    parser.add_argument("--text_device", default=None)
    parser.add_argument("--image_device", default="cuda")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.checkpoint_dir:
        config_path = os.path.join(args.checkpoint_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _update_dataclass(cfg, data)

    cfg.retrieval.index_dir = args.index_dir
    cfg.retrieval.image_index_path = f"{args.index_dir}/image.index"
    cfg.retrieval.image_meta_path = f"{args.index_dir}/image.meta.json"
    cfg.retrieval.image_embeds_path = f"{args.index_dir}/image.embeds.npy"
    cfg.retrieval.text_index_path = f"{args.index_dir}/text.index"
    cfg.retrieval.text_meta_path = f"{args.index_dir}/text.meta.json"
    cfg.retrieval.text_embeds_path = f"{args.index_dir}/text.embeds.npy"

    if args.dataset and args.dataset_prefixes:
        raise ValueError("Set either --dataset or --dataset_prefixes, not both.")

    if args.dataset or args.dataset_prefixes:
        prefixes = [args.dataset] if args.dataset else [
            p.strip() for p in args.dataset_prefixes.split(",") if p.strip()
        ]
        grouped = _load_grouped_samples(args.val_jsonl, prefixes, args.max_samples)
        datasets = grouped
    else:
        datasets = {"all": _load_samples(args.val_jsonl, args.max_samples)}

    text_retriever = None
    image_retriever = None
    if not args.skip_text:
        text_retriever = TextRetriever(cfg.retrieval.text_encoder_name, device=args.text_device)
        text_retriever.load(
            cfg.retrieval.text_index_path,
            cfg.retrieval.text_meta_path,
            cfg.retrieval.text_embeds_path,
        )
    if not args.skip_image:
        image_retriever = ImageRetriever(cfg.retrieval.image_encoder_name, device=args.image_device)
        image_retriever.load(
            cfg.retrieval.image_index_path,
            cfg.retrieval.image_meta_path,
            cfg.retrieval.image_embeds_path,
        )

    sim_thresholds = [float(x) for x in args.sim_thresholds.split(",") if x.strip()]
    overlap_thresholds = [
        float(x) for x in args.text_overlap_thresholds.split(",") if x.strip()
    ]
    hash_thresholds = [
        float(x) for x in args.hash_sim_thresholds.split(",") if x.strip()
    ]
    results: Dict[str, Any] = {
        "meta": {
            "top_k": args.top_k,
            "query_mode": args.query_mode,
            "sim_thresholds": sim_thresholds,
            "text_overlap_thresholds": overlap_thresholds,
            "hash_sim_thresholds": hash_thresholds,
        }
    }
    for name, samples in datasets.items():
        logging.info("Audit dataset=%s | samples=%d", name, len(samples))
        entry: Dict[str, Any] = {"count": len(samples)}
        if text_retriever is not None:
            entry["text"] = _audit_text(
                samples,
                text_retriever,
                args.top_k,
                args.query_mode,
                args.batch_size,
                sim_thresholds,
                overlap_thresholds,
            )
        if image_retriever is not None:
            entry["image"] = _audit_image(
                samples,
                image_retriever,
                args.top_k,
                args.image_root,
                args.batch_size,
                sim_thresholds,
                hash_thresholds,
            )
        results[name] = entry

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
