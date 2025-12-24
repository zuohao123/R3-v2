"""Entry point for training R3++."""
from __future__ import annotations

import argparse
import logging

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
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--index_dir", default="indices")
    parser.add_argument("--top_k", type=int, default=5)
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
    cfg.training.learning_rate = args.learning_rate
    cfg.training.gradient_accumulation = args.grad_accum
    cfg.model.use_lora = args.use_lora
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

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
