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
    parser.add_argument("--adam_eps", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--min_label_ratio", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--loss_scale", type=float, default=None)
    parser.add_argument("--disable_grad_scaler", action="store_true")
    parser.add_argument("--skip_nonfinite_grads", action="store_true")
    parser.add_argument("--lora_lr_mult", type=float, default=None)
    parser.add_argument("--r3_lr_mult", type=float, default=None)
    parser.add_argument("--gate_conf_weight", type=float, default=None)
    parser.add_argument("--gate_entropy_weight", type=float, default=None)
    parser.add_argument("--retrieval_align_weight", type=float, default=None)
    parser.add_argument("--retrieval_align_temperature", type=float, default=None)
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
    parser.add_argument(
        "--teacher_mode",
        choices=["copy", "ema", "shared"],
        default=None,
        help="Teacher mode: copy (full teacher), ema (EMA over trainable params), shared (no EMA).",
    )
    parser.add_argument("--teacher_ema_decay", type=float, default=None)
    parser.add_argument("--teacher_ema_update_steps", type=int, default=None)
    parser.add_argument("--teacher_ema_start_step", type=int, default=None)
    parser.add_argument("--disable_corruption", action="store_true")
    parser.add_argument("--max_corruption", type=float, default=None)
    parser.add_argument("--corruption_warmup_steps", type=int, default=None)
    parser.add_argument("--corruption_total_steps", type=int, default=None)
    parser.add_argument(
        "--corruption_schedule",
        choices=["linear", "cyclic"],
        default=None,
        help="Corruption schedule type (linear or cyclic).",
    )
    parser.add_argument("--corruption_cycles", type=int, default=None)
    parser.add_argument(
        "--corruption_max_severity",
        type=float,
        default=None,
        help="Scale corruption intensity (effective_level = level * max_severity).",
    )
    parser.add_argument("--disable_text_retrieval", action="store_true")
    parser.add_argument("--disable_image_retrieval", action="store_true")
    parser.add_argument("--disable_prefix", action="store_true")
    parser.add_argument("--disable_memory", action="store_true")
    parser.add_argument("--disable_visual_memory", action="store_true")
    parser.add_argument("--disable_latent_tokens", action="store_true")
    parser.add_argument("--disable_gate", action="store_true")
    parser.add_argument("--disable_context", action="store_true")
    parser.add_argument("--max_context_chars", type=int, default=None)
    parser.add_argument("--visual_memory_len", type=int, default=None)
    parser.add_argument("--use_soft_prefix", action="store_true")
    parser.add_argument("--disable_soft_prefix", action="store_true")
    parser.add_argument("--score_temperature", type=float, default=None)
    parser.add_argument("--min_text_score", type=float, default=None)
    parser.add_argument("--min_image_score", type=float, default=None)
    parser.add_argument("--max_text_score", type=float, default=None)
    parser.add_argument("--max_image_score", type=float, default=None)
    parser.add_argument("--disable_score_weighting", action="store_true")
    parser.add_argument("--r3_fp32", action="store_true", help="Run R3 modules in fp32")
    parser.add_argument("--r3_fp16", action="store_true", help="Allow autocast in R3 modules")
    parser.add_argument("--enable_router", action="store_true")
    parser.add_argument("--disable_router", action="store_true")
    parser.add_argument("--router_hidden", type=int, default=None)
    parser.add_argument("--router_dropout", type=float, default=None)
    parser.add_argument("--router_out_dim", type=int, default=None)
    parser.add_argument("--router_weight", type=float, default=None)
    parser.add_argument("--router_temperature", type=float, default=None)
    parser.add_argument("--router_warmup_steps", type=int, default=None)
    parser.add_argument("--train_router_only", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--resume_optimizer", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.resume_from:
        if not os.path.isdir(args.resume_from):
            raise FileNotFoundError(f"resume_from path not found: {args.resume_from}")
        r3_state = os.path.join(args.resume_from, "r3_state.pt")
        if not os.path.exists(r3_state):
            raise FileNotFoundError(
                f"resume_from missing r3_state.pt: {args.resume_from}"
            )
        if args.use_lora:
            adapter_safetensors = os.path.join(
                args.resume_from, "adapter_model.safetensors"
            )
            adapter_bin = os.path.join(args.resume_from, "adapter_model.bin")
            if not (os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin)):
                raise FileNotFoundError(
                    "resume_from missing LoRA adapter weights (adapter_model.safetensors|bin): "
                    f"{args.resume_from}"
                )

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
    if args.adam_eps is not None:
        cfg.training.adam_eps = args.adam_eps
    cfg.training.gradient_accumulation = args.grad_accum
    cfg.training.log_every = args.log_every
    if args.save_every is not None:
        cfg.training.save_every = args.save_every
    if args.eval_every is not None:
        cfg.training.eval_every = args.eval_every
    if args.max_grad_norm is not None:
        cfg.training.max_grad_norm = args.max_grad_norm
    if args.min_label_ratio is not None:
        cfg.training.min_label_ratio = args.min_label_ratio
    if args.max_length is not None:
        cfg.data.max_length = args.max_length
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.loss_scale is not None:
        cfg.training.loss_scale = args.loss_scale
    if args.disable_grad_scaler:
        cfg.training.disable_grad_scaler = True
    if args.skip_nonfinite_grads:
        cfg.training.skip_nonfinite_grads = True
    if args.lora_lr_mult is not None:
        cfg.training.lora_lr_mult = args.lora_lr_mult
    if args.r3_lr_mult is not None:
        cfg.training.r3_lr_mult = args.r3_lr_mult
    if args.gate_conf_weight is not None:
        cfg.loss.gate_conf_weight = args.gate_conf_weight
    if args.gate_entropy_weight is not None:
        cfg.loss.gate_entropy_weight = args.gate_entropy_weight
    if args.retrieval_align_weight is not None:
        cfg.loss.retrieval_align_weight = args.retrieval_align_weight
    if args.retrieval_align_temperature is not None:
        cfg.loss.retrieval_align_temperature = args.retrieval_align_temperature
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
    if args.teacher_mode is not None:
        cfg.training.teacher_mode = args.teacher_mode
    if args.teacher_ema_decay is not None:
        cfg.training.teacher_ema_decay = args.teacher_ema_decay
    if args.teacher_ema_update_steps is not None:
        cfg.training.teacher_ema_update_steps = args.teacher_ema_update_steps
    if args.teacher_ema_start_step is not None:
        cfg.training.teacher_ema_start_step = args.teacher_ema_start_step
    cfg.r3.enable_corruption = not args.disable_corruption
    if args.corruption_max_severity is not None:
        cfg.r3.corruption.max_severity = args.corruption_max_severity
    if args.max_corruption is not None:
        cfg.curriculum.max_corruption = args.max_corruption
    if args.corruption_warmup_steps is not None:
        cfg.curriculum.warmup_steps = args.corruption_warmup_steps
    if args.corruption_total_steps is not None:
        cfg.curriculum.total_steps = args.corruption_total_steps
    if args.corruption_schedule is not None:
        cfg.curriculum.schedule = args.corruption_schedule
    if args.corruption_cycles is not None:
        cfg.curriculum.cycles = args.corruption_cycles
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
    if args.router_weight is not None:
        cfg.loss.router_weight = args.router_weight
    if args.router_temperature is not None:
        cfg.loss.router_temperature = args.router_temperature
    if args.router_warmup_steps is not None:
        cfg.training.router_warmup_steps = args.router_warmup_steps
    if args.train_router_only:
        cfg.training.train_router_only = True
    if args.visual_memory_len is not None:
        cfg.r3.visual_memory_len = args.visual_memory_len
    if args.use_soft_prefix:
        cfg.r3.use_soft_prefix = True
    if args.disable_soft_prefix:
        cfg.r3.use_soft_prefix = False
    if args.score_temperature is not None:
        cfg.r3.score_temperature = args.score_temperature
    if args.min_text_score is not None:
        cfg.r3.min_text_score = args.min_text_score
    if args.min_image_score is not None:
        cfg.r3.min_image_score = args.min_image_score
    if args.max_text_score is not None:
        cfg.r3.max_text_score = args.max_text_score
    if args.max_image_score is not None:
        cfg.r3.max_image_score = args.max_image_score
    if args.disable_score_weighting:
        cfg.r3.use_score_weighting = False
    if args.r3_fp16:
        cfg.r3.force_fp32 = False
    if args.r3_fp32:
        cfg.r3.force_fp32 = True
    if args.resume_from:
        cfg.training.resume_from = args.resume_from
    if args.resume_optimizer:
        cfg.training.resume_optimizer = True

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
