"""Entry point for training R3++."""
from __future__ import annotations

import argparse
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.train_config import TrainConfig
from training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train R3++ model")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument(
        "--model_name",
        default=None,
        help="Local path or HF model id (e.g. models/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--min_label_ratio", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--sample_every", type=int, default=0)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--sample_max_new_tokens", type=int, default=32)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--sampling_alpha",
        type=float,
        default=None,
        help="Temperature sampling alpha in [0,1]; smaller = more balanced",
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument(
        "--lora_targets",
        default=None,
        help="Comma-separated target modules for LoRA (e.g., q_proj,k_proj,v_proj,o_proj)",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--index_dir", default="indices")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--backend",
        choices=["none", "fsdp", "deepspeed"],
        default="none",
        help="Distributed backend",
    )
    parser.add_argument("--deepspeed_config", default=None)
    parser.add_argument("--fsdp_min_params", type=int, default=100_000_000)
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument(
        "--fsdp_sharding",
        choices=["full", "grad", "no_shard"],
        default="full",
    )
    parser.add_argument("--disable_teacher", action="store_true")
    parser.add_argument("--disable_corruption", action="store_true")
    parser.add_argument("--disable_text_retrieval", action="store_true")
    parser.add_argument("--disable_image_retrieval", action="store_true")
    parser.add_argument("--disable_prefix", action="store_true")
    parser.add_argument("--disable_memory", action="store_true")
    parser.add_argument("--disable_latent_tokens", action="store_true")
    parser.add_argument("--disable_gate", action="store_true")
    parser.add_argument("--disable_context", action="store_true")
    parser.add_argument("--max_context_chars", type=int, default=None)
    parser.add_argument("--r3_fp32", action="store_true", help="Run R3 modules in fp32")
    parser.add_argument("--r3_fp16", action="store_true", help="Allow autocast in R3 modules")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = TrainConfig()
    cfg.data.train_jsonl = args.train_jsonl
    cfg.data.val_jsonl = args.val_jsonl
    cfg.data.image_root = args.image_root
    cfg.training.output_dir = args.output_dir
    if args.model_name:
        cfg.model.model_name = args.model_name
    cfg.training.batch_size = args.batch_size
    cfg.training.num_epochs = args.num_epochs
    cfg.training.max_steps = args.max_steps
    cfg.training.learning_rate = args.learning_rate
    cfg.training.gradient_accumulation = args.grad_accum
    cfg.training.log_every = args.log_every
    if args.max_grad_norm is not None:
        cfg.training.max_grad_norm = args.max_grad_norm
    if args.min_label_ratio is not None:
        cfg.training.min_label_ratio = args.min_label_ratio
    if args.max_length is not None:
        cfg.data.max_length = args.max_length
    cfg.training.sample_every = args.sample_every
    cfg.training.sample_num = args.sample_num
    cfg.training.sample_max_new_tokens = args.sample_max_new_tokens
    cfg.training.gradient_checkpointing = args.gradient_checkpointing
    cfg.training.sampling_alpha = args.sampling_alpha
    cfg.model.use_lora = args.use_lora
    if args.lora_targets:
        cfg.model.lora_target_modules = [
            name.strip() for name in args.lora_targets.split(",") if name.strip()
        ]
    cfg.training.bf16 = args.bf16 or cfg.training.bf16
    cfg.training.fp16 = args.fp16
    if args.fp16:
        cfg.model.torch_dtype = "fp16"
    if args.bf16:
        cfg.model.torch_dtype = "bf16"

    cfg.retrieval.index_dir = args.index_dir
    cfg.retrieval.image_index_path = f"{args.index_dir}/image.index"
    cfg.retrieval.image_meta_path = f"{args.index_dir}/image.meta.json"
    cfg.retrieval.image_embeds_path = f"{args.index_dir}/image.embeds.npy"
    cfg.retrieval.text_index_path = f"{args.index_dir}/text.index"
    cfg.retrieval.text_meta_path = f"{args.index_dir}/text.meta.json"
    cfg.retrieval.text_embeds_path = f"{args.index_dir}/text.embeds.npy"
    cfg.retrieval.top_k = args.top_k
    cfg.training.distributed_backend = args.backend
    cfg.training.deepspeed_config = args.deepspeed_config
    cfg.training.fsdp_min_num_params = args.fsdp_min_params
    cfg.training.fsdp_cpu_offload = args.fsdp_cpu_offload
    cfg.training.fsdp_sharding = args.fsdp_sharding
    cfg.training.use_teacher = not args.disable_teacher
    cfg.r3.enable_corruption = not args.disable_corruption
    cfg.r3.enable_text_retrieval = not args.disable_text_retrieval
    cfg.r3.enable_image_retrieval = not args.disable_image_retrieval
    cfg.r3.enable_prefix = not args.disable_prefix
    cfg.r3.enable_memory = not args.disable_memory
    cfg.r3.enable_latent_tokens = not args.disable_latent_tokens
    cfg.r3.enable_gate = not args.disable_gate
    cfg.r3.enable_context = not args.disable_context
    if args.max_context_chars is not None:
        cfg.r3.max_context_chars = args.max_context_chars
    if args.r3_fp16:
        cfg.r3.force_fp32 = False
    if args.r3_fp32:
        cfg.r3.force_fp32 = True

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
