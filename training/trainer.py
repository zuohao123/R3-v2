"""Trainer for R3++ multimodal QA."""
from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import os
import time
import random
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import functools
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional progress bar
    tqdm = None

from config.train_config import TrainConfig
from data.datasets import UnifiedQACollator, UnifiedQADataset
from models.qwen_wrapper import QwenVLConfig, QwenVLWrapper
from models.r3_modules import R3
from retrieval.image_retrieval import ImageRetriever
from retrieval.text_retrieval import TextRetriever
from training.curriculum import CurriculumScheduler
from training.losses import compute_total_loss, per_sample_cross_entropy


class R3TrainModule(torch.nn.Module):
    """Trainable module wrapping Qwen and R3++ for DeepSpeed."""

    def __init__(self, qwen: QwenVLWrapper, r3: R3) -> None:
        super().__init__()
        self.qwen_model = qwen.model
        self.r3 = r3
        self.qwen = qwen

    def forward(  # type: ignore[override]
        self,
        images,
        questions,
        pseudo_texts,
        answers,
        corruption_level: float,
        top_k: int,
        max_length: Optional[int] = None,
    ) -> Any:
        bundle = self.r3.forward_student(
            images,
            questions,
            pseudo_texts,
            answers,
            corruption_level,
            top_k,
            max_length=max_length,
        )
        return bundle


class EmaTeacher:
    """EMA helper for trainable parameters."""

    def __init__(
        self, model: torch.nn.Module, decay: float = 0.999, use_fp32: bool = True
    ) -> None:
        self.decay = decay
        self.use_fp32 = use_fp32
        self.params: list[tuple[str, torch.nn.Parameter]] = []
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad or not torch.is_floating_point(param):
                continue
            self.params.append((name, param))
            dtype = torch.float32 if use_fp32 else param.dtype
            self.shadow[name] = param.detach().to(dtype=dtype).clone()

    def update(self) -> None:
        if not self.params:
            return
        decay = self.decay
        for name, param in self.params:
            data = param.detach()
            if self.use_fp32:
                data = data.float()
            self.shadow[name].mul_(decay).add_(data, alpha=1.0 - decay)

    @contextlib.contextmanager
    def swap_to_ema(self):
        if not self.params:
            yield
            return
        backups: dict[str, torch.Tensor] = {}
        for name, param in self.params:
            backups[name] = param.data.clone()
            ema = self.shadow[name]
            if ema.device != param.device:
                ema = ema.to(param.device)
            if ema.dtype != param.dtype:
                ema = ema.to(param.dtype)
            param.data.copy_(ema)
        try:
            yield
        finally:
            for name, param in self.params:
                param.data.copy_(backups[name])


class DistributedTemperatureSampler(Sampler[int]):
    """Distributed weighted sampler with temperature smoothing."""

    def __init__(
        self,
        weights: torch.Tensor,
        num_replicas: int,
        rank: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        if self.drop_last:
            self.num_samples = len(self.weights) // self.num_replicas
        else:
            self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        ).tolist()
        indices = indices[: self.total_size]
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples


class Trainer:
    """Main training loop for R3++."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self._init_distributed()
        self.device = torch.device(
            config.model.device if torch.cuda.is_available() else "cpu"
        )
        self._set_seed(config.training.seed + self.rank)
        self.last_grad_norm: Optional[float] = None
        self.last_grad_nonfinite: Optional[float] = None
        self.last_grad_max_abs: Optional[float] = None
        self.last_bad_params: Optional[list[str]] = None

        self.train_dataset = UnifiedQADataset(
            config.data.train_jsonl,
            image_root=config.data.image_root,
            max_samples=config.data.max_samples,
        )
        self.val_dataset = UnifiedQADataset(
            config.data.val_jsonl,
            image_root=config.data.image_root,
            max_samples=config.evaluation.max_eval_samples,
        )

        qwen_cfg = QwenVLConfig(
            model_name=config.model.model_name,
            torch_dtype=config.model.torch_dtype,
            device=config.model.device,
            use_teacher=config.training.use_teacher,
            teacher_mode=config.training.teacher_mode,
            use_lora=config.model.use_lora,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            lora_target_modules=config.model.lora_target_modules,
        )
        self.qwen = QwenVLWrapper(qwen_cfg)
        if self.config.training.gradient_checkpointing:
            if hasattr(self.qwen.model, "gradient_checkpointing_enable"):
                try:
                    self.qwen.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    self.qwen.model.gradient_checkpointing_enable()
            if hasattr(self.qwen.model, "config"):
                self.qwen.model.config.use_cache = False
            self.qwen.enable_input_require_grads()
            if self.is_main_process:
                logging.info("Enabled gradient checkpointing.")

        self.text_retriever: Optional[TextRetriever] = None
        self.image_retriever: Optional[ImageRetriever] = None
        if config.retrieval.use_retrieval:
            self.text_retriever = TextRetriever(config.retrieval.text_encoder_name)
            self.image_retriever = ImageRetriever(
                config.retrieval.image_encoder_name, device=config.model.device
            )
            self.text_retriever.load(
                config.retrieval.text_index_path,
                config.retrieval.text_meta_path,
                config.retrieval.text_embeds_path,
            )
            self.image_retriever.load(
                config.retrieval.image_index_path,
                config.retrieval.image_meta_path,
                config.retrieval.image_embeds_path,
            )

        self.r3 = R3(self.qwen, self.text_retriever, self.image_retriever, config.r3)
        self.r3.to(self.device)
        # Clean any non-finite params (e.g., from resume or prior runs).
        with torch.no_grad():
            self.r3.sanitize_parameters()
        self._resume_from_checkpoint()
        if self.config.training.train_router_only:
            if self.r3.router is None:
                raise ValueError("train_router_only requires enable_router=True")
            for param in self.qwen.model.parameters():
                param.requires_grad = False
            for param in self.r3.parameters():
                param.requires_grad = False
            for param in self.r3.router.parameters():
                param.requires_grad = True
            if self.is_main_process:
                logging.info("Training router only (backbone and R3 modules frozen).")

        self.engine = None
        if self.config.training.distributed_backend == "fsdp":
            self._wrap_fsdp()
        elif self.config.training.distributed_backend == "deepspeed":
            self._init_deepspeed()

        self.teacher_mode = self.config.training.teacher_mode
        self.teacher_ema: Optional[EmaTeacher] = None
        self.optim_step = 0
        if self.config.training.use_teacher and self.teacher_mode == "ema":
            self.teacher_ema = EmaTeacher(
                self.qwen.model, decay=self.config.training.teacher_ema_decay, use_fp32=True
            )
            if self.is_main_process:
                logging.info(
                    "Initialized EMA teacher with %d trainable params.",
                    len(self.teacher_ema.params),
                )

        self.curriculum = CurriculumScheduler(
            max_corruption=config.curriculum.max_corruption,
            warmup_steps=config.curriculum.warmup_steps,
            total_steps=config.curriculum.total_steps,
            schedule=config.curriculum.schedule,
            cycles=config.curriculum.cycles,
        )
        if config.training.max_steps is not None:
            if self.curriculum.total_steps <= 0 or self.curriculum.total_steps > config.training.max_steps:
                self.curriculum.total_steps = config.training.max_steps
            if self.curriculum.warmup_steps > self.curriculum.total_steps:
                self.curriculum.warmup_steps = self.curriculum.total_steps

        if self.config.training.distributed_backend != "deepspeed":
            base_lr = config.training.learning_rate
            lora_lr = base_lr * config.training.lora_lr_mult
            r3_lr = base_lr * config.training.r3_lr_mult
            lora_params = []
            qwen_params = []
            for name, param in self.qwen.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "lora_" in name:
                    lora_params.append(param)
                else:
                    qwen_params.append(param)
            r3_params = [p for p in self.r3.parameters() if p.requires_grad]
            param_groups = []
            if lora_params:
                param_groups.append({"params": lora_params, "lr": lora_lr})
            if qwen_params:
                param_groups.append({"params": qwen_params, "lr": base_lr})
            if r3_params:
                param_groups.append({"params": r3_params, "lr": r3_lr})
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=config.training.weight_decay,
                eps=config.training.adam_eps,
            )
            if self.config.training.resume_from and self.config.training.resume_optimizer:
                opt_path = os.path.join(
                    self.config.training.resume_from, "optimizer.pt"
                )
                if os.path.exists(opt_path):
                    try:
                        self.optimizer.load_state_dict(
                            torch.load(opt_path, map_location="cpu")
                        )
                        if self.is_main_process:
                            logging.info("Loaded optimizer state from %s", opt_path)
                    except Exception as exc:
                        if self.is_main_process:
                            logging.warning(
                                "Failed to load optimizer state from %s: %s",
                                opt_path,
                                exc,
                            )
        else:
            self.optimizer = None

        if self.config.training.distributed_backend == "deepspeed" or self.config.training.disable_grad_scaler:
            self.scaler = None
        else:
            if config.training.fp16 and self.distributed:
                self.scaler = ShardedGradScaler(init_scale=config.training.loss_scale)
            elif config.training.fp16:
                self.scaler = torch.cuda.amp.GradScaler(init_scale=config.training.loss_scale)
            else:
                self.scaler = None

        self.train_sampler = None
        self.val_sampler = None
        sampling_alpha = config.training.sampling_alpha
        if sampling_alpha is not None:
            weights = self._build_sampling_weights(self.train_dataset, sampling_alpha)
            if self.distributed:
                self.train_sampler = DistributedTemperatureSampler(
                    weights,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    seed=config.training.seed,
                )
            else:
                self.train_sampler = WeightedRandomSampler(
                    weights,
                    num_samples=len(weights),
                    replacement=True,
                )
        elif self.distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        if self.distributed:
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=config.data.num_workers,
            collate_fn=UnifiedQACollator(
                tokenizer=self.qwen.tokenizer,
                max_length=config.data.max_length,
                image_root=config.data.image_root,
                image_size=config.data.image_size,
            ),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.evaluation.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=config.data.num_workers,
            collate_fn=UnifiedQACollator(
                tokenizer=self.qwen.tokenizer,
                max_length=config.data.max_length,
                image_root=config.data.image_root,
                image_size=config.data.image_size,
            ),
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _dataset_name(image_path: str) -> str:
        if not image_path:
            return "unknown"
        parts = image_path.split("/")
        return parts[0] if parts else "unknown"

    def _build_sampling_weights(
        self, dataset: UnifiedQADataset, alpha: float
    ) -> torch.Tensor:
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("sampling_alpha must be in [0, 1].")
        counts: Dict[str, int] = {}
        for item in dataset.samples:
            name = self._dataset_name(item.image_path)
            counts[name] = counts.get(name, 0) + 1
        weights = []
        for item in dataset.samples:
            name = self._dataset_name(item.image_path)
            count = max(1, counts.get(name, 1))
            weight = count ** (alpha - 1.0)
            weights.append(weight)
        weights_tensor = torch.tensor(weights, dtype=torch.double)
        if self.is_main_process:
            logging.info("Temperature sampling alpha=%.2f, dataset counts=%s", alpha, counts)
        return weights_tensor

    def _init_distributed(self) -> None:
        self.distributed = False
        self.rank = 0
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.is_main_process = True

        backend = self.config.training.distributed_backend
        if backend == "none" and self.world_size > 1:
            logging.warning(
                "WORLD_SIZE=%d but backend=none; defaulting to fsdp.",
                self.world_size,
            )
            backend = "fsdp"
            self.config.training.distributed_backend = backend
        if backend != "none" and self.world_size == 1:
            logging.warning(
                "Distributed backend %s requested but WORLD_SIZE=1; falling back to single process.",
                backend,
            )
            self.config.training.distributed_backend = "none"
            backend = "none"

        if backend == "none":
            self.config.model.device = "cuda" if torch.cuda.is_available() else "cpu"
            return

        if not torch.cuda.is_available():
            raise RuntimeError("Distributed backend requires CUDA.")

        torch.cuda.set_device(self.local_rank)
        self.config.model.device = f"cuda:{self.local_rank}"
        self.distributed = True
        if backend == "deepspeed":
            import deepspeed

            deepspeed.init_distributed()
        else:
            dist.init_process_group(backend="nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main_process = self.rank == 0

    def _wrap_fsdp(self) -> None:
        if not self.distributed:
            return
        dtype = torch.float32
        if self.config.training.fp16:
            dtype = torch.float16
        elif self.config.training.bf16:
            dtype = torch.bfloat16
        mp_policy = MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
        )
        cpu_offload = CPUOffload(offload_params=self.config.training.fsdp_cpu_offload)
        sharding_map = {
            "full": ShardingStrategy.FULL_SHARD,
            "grad": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        sharding = sharding_map.get(self.config.training.fsdp_sharding, ShardingStrategy.FULL_SHARD)
        auto_wrap_policy = None
        wrap_classes = set()
        for module in self.qwen.model.modules():
            name = module.__class__.__name__
            if any(key in name for key in ("DecoderLayer", "TransformerLayer", "Block")):
                wrap_classes.add(module.__class__)
        if wrap_classes:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=wrap_classes,
            )
        elif self.config.training.fsdp_min_num_params > 0:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=self.config.training.fsdp_min_num_params,
            )
        ignored_modules = []
        if self.config.r3.use_soft_prefix:
            try:
                emb = self.qwen.model.get_input_embeddings()
                if emb is not None:
                    ignored_modules.append(emb)
            except Exception:
                if self.is_main_process:
                    logging.warning(
                        "Failed to resolve input embeddings; disabling FSDP auto-wrap."
                    )
                auto_wrap_policy = None
        self.qwen.model = FSDP(
            self.qwen.model,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding,
            use_orig_params=self.config.training.fsdp_use_orig_params,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.device,
            ignored_modules=ignored_modules,
            limit_all_gathers=True,
        )

    def _build_deepspeed_config(self) -> Dict[str, Any]:
        ds_config: Dict[str, Any] = {}
        if self.config.training.deepspeed_config and os.path.exists(
            self.config.training.deepspeed_config
        ):
            with open(self.config.training.deepspeed_config, "r", encoding="utf-8") as f:
                ds_config = json.load(f)
        else:
            ds_config = {
                "zero_optimization": {
                    "stage": 3,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "contiguous_gradients": True,
                    "stage3_param_persistence_threshold": 100000,
                },
                "gradient_clipping": 1.0,
            }

        ds_config.setdefault(
            "optimizer",
            {
                "type": "AdamW",
                "params": {
                    "lr": self.config.training.learning_rate,
                    "weight_decay": self.config.training.weight_decay,
                },
            },
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.config.training.batch_size
        ds_config["gradient_accumulation_steps"] = self.config.training.gradient_accumulation
        ds_config["train_batch_size"] = (
            self.config.training.batch_size
            * self.config.training.gradient_accumulation
            * max(1, self.world_size)
        )
        ds_config["fp16"] = {"enabled": bool(self.config.training.fp16)}
        ds_config["bf16"] = {"enabled": bool(self.config.training.bf16)}
        if self.config.training.max_grad_norm and self.config.training.max_grad_norm > 0:
            ds_config.setdefault("gradient_clipping", self.config.training.max_grad_norm)
        return ds_config

    def _init_deepspeed(self) -> None:
        import deepspeed

        self.train_module = R3TrainModule(self.qwen, self.r3)
        ds_config = self._build_deepspeed_config()
        model_parameters = [p for p in self.train_module.parameters() if p.requires_grad]
        engine, optimizer, _, _ = deepspeed.initialize(
            model=self.train_module,
            model_parameters=model_parameters,
            config=ds_config,
        )
        self.engine = engine
        self.optimizer = optimizer

    def _maybe_update_ema(self) -> None:
        if self.teacher_ema is None:
            return
        if self.optim_step < self.config.training.teacher_ema_start_step:
            return
        if self.config.training.teacher_ema_update_steps <= 0:
            return
        if self.optim_step % self.config.training.teacher_ema_update_steps != 0:
            return
        with torch.no_grad():
            self.teacher_ema.update()

    def _forward_teacher_shared(
        self,
        images,
        questions,
        pseudo_texts,
        answers,
    ) -> Any:
        was_training = self.qwen.model.training
        self.qwen.model.eval()
        with torch.no_grad():
            with self._autocast():
                outputs = self.qwen.forward_student(
                    images,
                    questions,
                    pseudo_texts,
                    answers,
                    max_length=self.config.data.max_length,
                )
        if was_training:
            self.qwen.model.train()
        return outputs

    def _forward_teacher_ema(
        self,
        images,
        questions,
        pseudo_texts,
        answers,
    ) -> Any:
        if self.teacher_ema is None:
            return None
        was_training = self.qwen.model.training
        self.qwen.model.eval()
        with torch.no_grad():
            with self.teacher_ema.swap_to_ema():
                with self._autocast():
                    outputs = self.qwen.forward_student(
                        images,
                        questions,
                        pseudo_texts,
                        answers,
                        max_length=self.config.data.max_length,
                    )
        if was_training:
            self.qwen.model.train()
        return outputs

    def _resume_from_checkpoint(self) -> None:
        resume_dir = self.config.training.resume_from
        if not resume_dir:
            return
        if not os.path.isdir(resume_dir):
            if self.is_main_process:
                logging.warning("Resume path %s is not a directory.", resume_dir)
            return
        if self.config.model.use_lora and self.config.training.resume_lora:
            try:
                loaded = self.qwen.load_lora_adapter(resume_dir)
            except Exception as exc:
                loaded = False
                if self.is_main_process:
                    logging.warning("Failed to load LoRA adapter: %s", exc)
            if self.is_main_process and loaded:
                logging.info("Loaded LoRA adapter from %s", resume_dir)
        r3_path = os.path.join(resume_dir, "r3_state.pt")
        if self.config.training.resume_r3 and os.path.exists(r3_path):
            try:
                state = torch.load(r3_path, map_location="cpu")
                missing, unexpected = self.r3.load_state_dict(state, strict=False)
                if self.is_main_process:
                    logging.info("Loaded R3 state from %s", r3_path)
                    if missing:
                        logging.info("R3 missing keys: %s", missing)
                    if unexpected:
                        logging.info("R3 unexpected keys: %s", unexpected)
            except Exception as exc:
                if self.is_main_process:
                    logging.warning("Failed to load R3 state: %s", exc)

    def _autocast(self):
        use_amp = self.config.training.fp16 or self.config.training.bf16
        if self.config.training.distributed_backend == "deepspeed" or not use_amp:
            return torch.autocast(device_type=self.device.type, enabled=False)
        dtype = torch.float16 if self.config.training.fp16 else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=True)

    def _save_checkpoint(self, step: int) -> None:
        out_dir = os.path.join(self.config.training.output_dir, f"step_{step}")
        os.makedirs(out_dir, exist_ok=True)
        if self.is_main_process:
            logging.info("Saving checkpoint to %s", out_dir)
        if self.config.training.distributed_backend == "fsdp":
            state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.qwen.model, StateDictType.FULL_STATE_DICT, state_cfg
            ):
                full_state = self.qwen.model.state_dict()
            if self.is_main_process:
                self.qwen.model.module.save_pretrained(out_dir, state_dict=full_state)
        elif self.config.training.distributed_backend == "deepspeed":
            assert self.engine is not None
            self.engine.save_checkpoint(out_dir)
            if not self.is_main_process:
                return
        else:
            if self.is_main_process:
                self.qwen.model.save_pretrained(out_dir)

        if self.is_main_process:
            self.qwen.processor.save_pretrained(out_dir)
            torch.save(self.r3.state_dict(), os.path.join(out_dir, "r3_state.pt"))
            if self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), os.path.join(out_dir, "optimizer.pt"))
            with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2)
            logging.info("Saved checkpoint to %s", out_dir)
        if self.distributed:
            dist.barrier()

    def _clip_gradients(self) -> None:
        max_norm = self.config.training.max_grad_norm
        if not max_norm or max_norm <= 0:
            return
        if self.config.training.distributed_backend == "fsdp":
            norm_qwen = FSDP.clip_grad_norm_(self.qwen.model, max_norm)
            norm_r3 = torch.nn.utils.clip_grad_norm_(self.r3.parameters(), max_norm)
            try:
                norm_val = float(torch.max(norm_qwen, norm_r3).item())
            except Exception:
                norm_val = float(norm_qwen) if hasattr(norm_qwen, "item") else None
            self.last_grad_norm = norm_val
            return
        params = [
            p
            for p in itertools.chain(self.qwen.model.parameters(), self.r3.parameters())
            if p.requires_grad
        ]
        if params:
            norm_val = torch.nn.utils.clip_grad_norm_(params, max_norm)
            self.last_grad_norm = float(norm_val)

    @staticmethod
    def _logit_stats(logits: torch.Tensor, max_tokens: int = 8, max_vocab: int = 2048) -> Dict[str, float]:
        """Compute lightweight diagnostic stats from a logit slice."""
        if logits is None:
            return {}
        seq_len = min(logits.size(1), max_tokens)
        vocab = min(logits.size(-1), max_vocab)
        raw = logits[:, :seq_len, :vocab].float()
        nan_count = float(torch.isnan(raw).sum().item())
        inf_count = float(torch.isinf(raw).sum().item())
        sample = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "logit_max": float(sample.max().item()),
            "logit_min": float(sample.min().item()),
            "logit_std": float(sample.std().item()),
            "logit_abs": float(sample.abs().mean().item()),
            "logit_nan": nan_count,
            "logit_inf": inf_count,
        }

    def _grad_finite_stats(self) -> Dict[str, float]:
        """Inspect gradients for non-finite values."""
        bad_params: list[str] = []
        bad_params_full: list[str] = []
        params = itertools.chain(
            self.qwen.model.named_parameters(), self.r3.named_parameters()
        )
        nonfinite = 0
        max_abs = 0.0
        for name, param in params:
            grad = param.grad
            if grad is None:
                continue
            if not torch.isfinite(grad).all():
                nonfinite += 1
                grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                bad_params_full.append(name)
                if len(bad_params) < 3:
                    bad_params.append(name)
            max_abs = max(max_abs, float(grad.abs().max().item()))
        self.last_grad_nonfinite = float(nonfinite)
        self.last_grad_max_abs = float(max_abs)
        if bad_params:
            self.last_bad_params = bad_params
        if bad_params_full:
            aux_only = True
            for name in bad_params_full:
                if (
                    "memory_aligner" in name
                    or "latent_tokens" in name
                    or "prefix_enhancer" in name
                    or "visual_memory" in name
                ):
                    continue
                aux_only = False
                break
            self.last_bad_in_aux_only = aux_only
        else:
            self.last_bad_in_aux_only = False
        return {"grad_nonfinite": float(nonfinite), "grad_max_abs": float(max_abs)}

    @staticmethod
    def _truncate(text: str, max_len: int = 240) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _log_samples(self, batch: Dict[str, Any], corruption_level: float, step: int) -> None:
        sample_every = self.config.training.sample_every
        if sample_every <= 0:
            return
        if self.config.training.distributed_backend == "deepspeed":
            if step == 0 and self.is_main_process:
                logging.warning(
                    "Sample logging is disabled under %s.",
                    self.config.training.distributed_backend,
                )
            return
        if step % sample_every != 0:
            return

        clean = batch["clean"]
        corrupted = batch["corrupted"]
        n = min(self.config.training.sample_num, len(clean["questions"]))
        if n <= 0:
            return
        images = corrupted["images"][:n]
        image_paths = corrupted.get("image_paths", [""] * n)[:n]
        questions = corrupted["questions"][:n]
        pseudo_texts = corrupted["pseudo_texts"][:n]
        answers = clean["answers"][:n]

        self.qwen.model.eval()
        self.r3.eval()
        if self.distributed and self.config.training.distributed_backend == "fsdp":
            with FSDP.summon_full_params(
                self.qwen.model, rank0_only=True, writeback=False
            ):
                if not self.is_main_process:
                    self.qwen.model.train()
                    self.r3.train()
                    return
                with torch.no_grad():
                    outputs = self.r3.generate(
                        images,
                        questions,
                        pseudo_texts,
                        corruption_level=corruption_level,
                        top_k=self.config.retrieval.top_k,
                        max_new_tokens=self.config.training.sample_max_new_tokens,
                        return_retrieval=True,
                    )
        else:
            with torch.no_grad():
                outputs = self.r3.generate(
                    images,
                    questions,
                    pseudo_texts,
                    corruption_level=corruption_level,
                    top_k=self.config.retrieval.top_k,
                    max_new_tokens=self.config.training.sample_max_new_tokens,
                    return_retrieval=True,
                )
        preds, retrieved_texts, retrieved_image_paths, contexts, prompts = outputs
        self.qwen.model.train()
        self.r3.train()

        if self.is_main_process:
            logging.info(
                "Sample outputs at step %d (corruption=%.2f):",
                step,
                corruption_level,
            )
            for idx in range(n):
                logging.info("  DATA image_path: %s", image_paths[idx])
                logging.info("  DATA question: %s", self._truncate(questions[idx]))
                logging.info("  DATA answer(gt): %s", self._truncate(answers[idx]))
                logging.info("  DATA pseudo_text: %s", self._truncate(pseudo_texts[idx]))
                if contexts[idx]:
                    logging.info("  R3 context: %s", self._truncate(contexts[idx]))
                if retrieved_texts[idx]:
                    shown = retrieved_texts[idx][: min(3, len(retrieved_texts[idx]))]
                    joined = " || ".join(self._truncate(t) for t in shown if t)
                    if joined:
                        logging.info("  RETRIEVAL text: %s", joined)
                if retrieved_image_paths[idx]:
                    shown_imgs = retrieved_image_paths[idx][: min(3, len(retrieved_image_paths[idx]))]
                    joined_imgs = " || ".join(self._truncate(p) for p in shown_imgs if p)
                    if joined_imgs:
                        logging.info("  RETRIEVAL images: %s", joined_imgs)
                if prompts[idx]:
                    logging.info("  MODEL input_text: %s", self._truncate(prompts[idx], 600))
                logging.info("  MODEL pred: %s", self._truncate(preds[idx]))

    def train(self) -> None:
        total_steps = self.config.training.max_steps
        if total_steps is None:
            try:
                total_steps = len(self.train_loader) * self.config.training.num_epochs
            except TypeError:
                total_steps = None
        global_step = 0
        start_time = time.time()
        pbar = None
        if self.is_main_process and tqdm is not None:
            pbar = tqdm(
                total=total_steps,
                desc="train",
                dynamic_ncols=True,
            )
        if self.engine is not None:
            self.engine.train()
        else:
            self.qwen.model.train()

        for epoch in range(self.config.training.num_epochs):
            if self.train_sampler is not None:
                if hasattr(self.train_sampler, "set_epoch"):
                    self.train_sampler.set_epoch(epoch)
            for step_in_epoch, batch in enumerate(self.train_loader):
                if total_steps is not None and global_step >= total_steps:
                    if pbar is not None:
                        pbar.close()
                    return

                corruption_level = self.curriculum.get_level(global_step)
                clean = batch["clean"]
                corrupted = batch["corrupted"]
                if any(not ans for ans in clean["answers"]):
                    clean["answers"] = [ans if ans else "unknown" for ans in clean["answers"]]

                teacher_outputs = None
                if self.config.training.use_teacher:
                    if self.teacher_mode == "ema":
                        teacher_outputs = self._forward_teacher_ema(
                            clean["images"],
                            clean["questions"],
                            corrupted["pseudo_texts"],
                            clean["answers"],
                        )
                    elif self.teacher_mode == "shared":
                        teacher_outputs = self._forward_teacher_shared(
                            clean["images"],
                            clean["questions"],
                            corrupted["pseudo_texts"],
                            clean["answers"],
                        )
                    else:
                        with self._autocast():
                            teacher_outputs = self.qwen.forward_teacher(
                                clean["images"],
                                clean["questions"],
                                corrupted["pseudo_texts"],
                                clean["answers"],
                                max_length=self.config.data.max_length,
                            )

                if self.config.training.distributed_backend == "deepspeed":
                    with self._autocast():
                        assert self.engine is not None
                        student_bundle = self.engine(
                            corrupted["images"],
                            corrupted["questions"],
                            corrupted["pseudo_texts"],
                            clean["answers"],
                            corruption_level,
                            self.config.retrieval.top_k,
                            self.config.data.max_length,
                        )
                        losses = compute_total_loss(
                            student_bundle,
                            teacher_outputs,
                            self.config.loss.consistency_weight,
                            self.config.loss.temperature,
                            loss_cfg=self.config.loss,
                        )
                        loss = losses["total"]
                    self.engine.backward(loss)
                    self.engine.step()
                    self.optim_step += 1
                    self._maybe_update_ema()
                else:
                    with self._autocast():
                        router_override = None
                        if self.config.r3.enable_router:
                            if (
                                self.config.training.train_router_only
                                or global_step < self.config.training.router_warmup_steps
                            ):
                                router_override = 1.0
                        student_bundle = self.r3.forward_student(
                            corrupted["images"],
                            corrupted["questions"],
                            corrupted["pseudo_texts"],
                            clean["answers"],
                            corruption_level,
                            self.config.retrieval.top_k,
                            max_length=self.config.data.max_length,
                            router_alpha_override=router_override,
                        )
                        losses = compute_total_loss(
                            student_bundle,
                            teacher_outputs,
                            self.config.loss.consistency_weight,
                            self.config.loss.temperature,
                            loss_cfg=self.config.loss,
                        )
                        loss = losses["total"] / self.config.training.gradient_accumulation
                    if (
                        self.config.r3.enable_router
                        and self.config.loss.router_weight > 0
                        and global_step >= self.config.training.router_warmup_steps
                    ):
                        def _with_seed(seed: int, override):
                            py_state = random.getstate()
                            np_state = np.random.get_state()
                            torch_state = torch.random.get_rng_state()
                            cuda_state = None
                            if torch.cuda.is_available():
                                cuda_state = torch.cuda.get_rng_state()
                            random.seed(seed)
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(seed)
                            try:
                                return self.r3.forward_student(
                                    corrupted["images"],
                                    corrupted["questions"],
                                    corrupted["pseudo_texts"],
                                    clean["answers"],
                                    corruption_level,
                                    self.config.retrieval.top_k,
                                    max_length=self.config.data.max_length,
                                    router_alpha_override=override,
                                )
                            finally:
                                random.setstate(py_state)
                                np.random.set_state(np_state)
                                torch.random.set_rng_state(torch_state)
                                if cuda_state is not None:
                                    torch.cuda.set_rng_state(cuda_state)

                        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                        out_dim = max(1, int(self.config.r3.router_out_dim))
                        override_full = (1.0, 1.0) if out_dim > 1 else 1.0
                        override_rag = (0.0, 0.0) if out_dim > 1 else 0.0
                        override_no_latent = (1.0, 0.0) if out_dim > 1 else 1.0
                        with torch.no_grad():
                            full_bundle = _with_seed(seed, override_full)
                            rag_bundle = _with_seed(seed, override_rag)
                            no_latent_bundle = (
                                _with_seed(seed, override_no_latent) if out_dim > 1 else None
                            )
                        full_ce = per_sample_cross_entropy(full_bundle["outputs"])
                        rag_ce = per_sample_cross_entropy(rag_bundle["outputs"])
                        logits = torch.stack([-rag_ce, -full_ce], dim=-1)
                        logits = logits / max(self.config.loss.router_temperature, 1e-6)
                        target = torch.softmax(logits, dim=-1)
                        p_full = target[:, 1].clamp(1e-4, 1.0 - 1e-4)
                        p_latent = p_full
                        if out_dim > 1 and no_latent_bundle is not None:
                            no_latent_ce = per_sample_cross_entropy(no_latent_bundle["outputs"])
                            logits_latent = torch.stack([-no_latent_ce, -full_ce], dim=-1)
                            logits_latent = logits_latent / max(
                                self.config.loss.router_temperature, 1e-6
                            )
                            target_latent = torch.softmax(logits_latent, dim=-1)
                            p_latent = target_latent[:, 1].clamp(1e-4, 1.0 - 1e-4)
                        alpha_pred = student_bundle.get("router_alpha")
                        if alpha_pred is not None:
                            if alpha_pred.dim() == 2 and alpha_pred.size(1) > 1:
                                alpha_prefix = alpha_pred[:, 0]
                                alpha_latent = alpha_pred[:, 1]
                                loss_prefix = F.binary_cross_entropy(alpha_prefix, p_full)
                                loss_latent = F.binary_cross_entropy(alpha_latent, p_latent)
                                router_loss = 0.5 * (loss_prefix + loss_latent)
                            else:
                                alpha_pred = alpha_pred.squeeze(-1)
                                router_loss = F.binary_cross_entropy(alpha_pred, p_full)
                            losses["router"] = router_loss
                            losses["total"] = losses["total"] + (
                                self.config.loss.router_weight * router_loss
                            )
                            loss = losses["total"] / self.config.training.gradient_accumulation

                    label_ratio_val = None
                    if "label_ratio" in losses:
                        label_ratio_val = float(losses["label_ratio"].item())
                    skip_local = False
                    if (
                        self.config.training.min_label_ratio > 0
                        and label_ratio_val is not None
                        and label_ratio_val < self.config.training.min_label_ratio
                    ):
                        skip_local = True
                    if not torch.isfinite(losses["total"]):
                        skip_local = True
                    if self.distributed:
                        skip_tensor = torch.tensor(
                            1 if skip_local else 0, device=self.device, dtype=torch.int
                        )
                        dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
                        skip_local = bool(skip_tensor.item() > 0)
                    if skip_local:
                        if self.is_main_process:
                            if label_ratio_val is not None and label_ratio_val < self.config.training.min_label_ratio:
                                logging.warning(
                                    "Low label ratio %.4f at step %d; skipping update.",
                                    label_ratio_val,
                                    global_step,
                                )
                            if not torch.isfinite(losses["total"]):
                                logging.warning(
                                    "Non-finite loss at step %d; skipping update.",
                                    global_step,
                                )
                        if self.optimizer is not None:
                            self.optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (global_step + 1) % self.config.training.gradient_accumulation == 0:
                        did_step = False
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                        grad_stats = self._grad_finite_stats()
                        local_nonfinite = grad_stats["grad_nonfinite"] > 0
                        local_nonaux = local_nonfinite and not getattr(
                            self, "last_bad_in_aux_only", False
                        )
                        global_nonfinite = local_nonfinite
                        global_nonaux = local_nonaux
                        if self.distributed:
                            flags = torch.tensor(
                                [1 if local_nonfinite else 0, 1 if local_nonaux else 0],
                                device=self.device,
                                dtype=torch.int,
                            )
                            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                            global_nonfinite = bool(flags[0].item() > 0)
                            global_nonaux = bool(flags[1].item() > 0)
                        skip_step = global_nonfinite
                        if skip_step and self.config.training.skip_nonfinite_grads:
                            if global_nonfinite and not global_nonaux:
                                if self.is_main_process:
                                    logging.warning(
                                        "Non-finite grads in auxiliary modules; zeroing and continuing."
                                    )
                                    if self.last_bad_params:
                                        logging.warning(
                                            "Non-finite grad params (sample): %s",
                                            ", ".join(self.last_bad_params),
                                        )
                                if self.r3.memory_aligner is not None:
                                    for param in self.r3.memory_aligner.parameters():
                                        if param.grad is not None:
                                            param.grad.data = torch.zeros_like(param.grad.data)
                                if self.r3.latent_tokens is not None:
                                    for param in self.r3.latent_tokens.parameters():
                                        if param.grad is not None:
                                            param.grad.data = torch.zeros_like(param.grad.data)
                                if self.r3.prefix_enhancer is not None:
                                    for param in self.r3.prefix_enhancer.parameters():
                                        if param.grad is not None:
                                            param.grad.data = torch.zeros_like(param.grad.data)
                                if self.r3.visual_memory is not None:
                                    for param in self.r3.visual_memory.parameters():
                                        if param.grad is not None:
                                            param.grad.data = torch.zeros_like(param.grad.data)
                                with torch.no_grad():
                                    self.r3.sanitize_parameters()
                                skip_step = False
                            else:
                                if self.is_main_process:
                                    logging.warning(
                                        "Non-finite grads detected; skipping optimizer step."
                                    )
                                    if self.last_bad_params:
                                        logging.warning(
                                            "Non-finite grad params (sample): %s",
                                            ", ".join(self.last_bad_params),
                                        )
                                with torch.no_grad():
                                    self.r3.sanitize_parameters()
                                if self.scaler:
                                    # Reset scaler state even when skipping the step to avoid double-unscale.
                                    self.scaler.update()
                                if self.optimizer is not None:
                                    self.optimizer.zero_grad(set_to_none=True)
                                global_step += 1
                                if pbar is not None:
                                    pbar.update(1)
                                continue
                        if skip_step and self.is_main_process:
                            logging.warning(
                                "Non-finite grads detected; zeroed and continuing."
                            )
                            if self.last_bad_params:
                                logging.warning(
                                    "Non-finite grad params (sample): %s",
                                    ", ".join(self.last_bad_params),
                                )
                        if self.scaler:
                            self._clip_gradients()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale = self.scaler.get_scale()
                            did_step = True
                            if self.distributed:
                                scale_flag = torch.tensor(
                                    1 if scale < 1 else 0,
                                    device=self.device,
                                    dtype=torch.int,
                                )
                                dist.all_reduce(scale_flag, op=dist.ReduceOp.MAX)
                                if scale_flag.item() > 0:
                                    scale = 0.0
                            if scale < 1:
                                if self.is_main_process:
                                    logging.warning(
                                        "Grad scaler scale dropped to %.3e; resetting.",
                                        scale,
                                    )
                                if self.distributed:
                                    self.scaler = ShardedGradScaler(
                                        init_scale=self.config.training.loss_scale
                                    )
                                else:
                                    self.scaler = torch.cuda.amp.GradScaler(
                                        init_scale=self.config.training.loss_scale
                                    )
                        else:
                            assert self.optimizer is not None
                            self._clip_gradients()
                            self.optimizer.step()
                            did_step = True
                        assert self.optimizer is not None
                        self.optimizer.zero_grad(set_to_none=True)
                        if did_step:
                            self.optim_step += 1
                            self._maybe_update_ema()

                if global_step % self.config.training.log_every == 0 and self.is_main_process:
                    lr = None
                    if self.optimizer is not None and self.optimizer.param_groups:
                        lr = self.optimizer.param_groups[0].get("lr")
                    scale = self.scaler.get_scale() if self.scaler is not None else None
                    gate_stats = None
                    conf_stats = None
                    logit_stats: Dict[str, float] = {}
                    if isinstance(student_bundle, dict) and "gates" in student_bundle:
                        gates = student_bundle["gates"].mean(dim=0).tolist()
                        gate_stats = tuple(round(g, 3) for g in gates)
                    if isinstance(student_bundle, dict) and "c_vis" in student_bundle:
                        c_vis = student_bundle["c_vis"].mean().item()
                        c_text = student_bundle["c_text"].mean().item()
                        conf_stats = (round(c_vis, 3), round(c_text, 3))
                    if isinstance(student_bundle, dict) and "outputs" in student_bundle:
                        outputs = student_bundle["outputs"]
                        if hasattr(outputs, "logits") and outputs.logits is not None:
                            logit_stats = self._logit_stats(outputs.logits)
                    elapsed = time.time() - start_time
                    eta = None
                    if total_steps is not None and global_step > 0:
                        rate = elapsed / global_step
                        eta = rate * max(total_steps - global_step, 0)
                    label_ratio = None
                    if "label_ratio" in losses:
                        label_ratio = float(losses["label_ratio"].item())
                    gate_conf = losses.get("gate_conf")
                    gate_ent = losses.get("gate_entropy")
                    r_align = losses.get("retrieval_align")
                    logging.info(
                        "epoch %d step %d | loss %.4f ce %.4f cons %.4f gconf %.4f gent %.4f ralign %.4f | corr %.2f | lr %s | scale %s | gate %s | conf %s | label %.3f | grad %s max_abs %s nonfinite %s | logit[max %.2f min %.2f std %.2f abs %.2f nan %.0f inf %.0f] | eta %s",
                        epoch,
                        global_step,
                        losses["total"].item(),
                        losses["ce"].item(),
                        losses["consistency"].item(),
                        gate_conf.item() if gate_conf is not None else 0.0,
                        gate_ent.item() if gate_ent is not None else 0.0,
                        r_align.item() if r_align is not None else 0.0,
                        corruption_level,
                        f"{lr:.2e}" if lr is not None else "n/a",
                        f"{scale:.1f}" if scale is not None else "n/a",
                        gate_stats if gate_stats is not None else "n/a",
                        conf_stats if conf_stats is not None else "n/a",
                        label_ratio if label_ratio is not None else -1.0,
                        f"{self.last_grad_norm:.2f}" if self.last_grad_norm is not None else "n/a",
                        f"{self.last_grad_max_abs:.2f}" if self.last_grad_max_abs is not None else "n/a",
                        f"{self.last_grad_nonfinite:.0f}" if self.last_grad_nonfinite is not None else "n/a",
                        logit_stats.get("logit_max", 0.0),
                        logit_stats.get("logit_min", 0.0),
                        logit_stats.get("logit_std", 0.0),
                        logit_stats.get("logit_abs", 0.0),
                        logit_stats.get("logit_nan", 0.0),
                        logit_stats.get("logit_inf", 0.0),
                        f"{eta/60:.1f}m" if eta is not None else "n/a",
                    )

                if pbar is not None:
                    postfix = {
                        "loss": f"{losses['total'].item():.3f}",
                        "corr": f"{corruption_level:.2f}",
                    }
                    if self.optimizer is not None and self.optimizer.param_groups:
                        postfix["lr"] = f"{self.optimizer.param_groups[0].get('lr'):.1e}"
                    pbar.set_postfix(postfix)
                    pbar.update(1)

                self._log_samples(batch, corruption_level, global_step)

                step_id = global_step + 1
                if step_id > 0 and step_id % self.config.training.eval_every == 0:
                    self.evaluate()

                if step_id > 0 and self.config.training.save_every:
                    if step_id % self.config.training.save_every == 0:
                        self._save_checkpoint(step_id)
                if total_steps is not None and step_id >= total_steps:
                    # Ensure a final checkpoint is written even if save_every
                    # is larger than max_steps or not aligned.
                    if not self.config.training.save_every or (
                        step_id % self.config.training.save_every != 0
                    ):
                        self._save_checkpoint(step_id)

                global_step += 1

        if pbar is not None:
            pbar.close()

    def evaluate(self) -> None:
        from evaluation.evaluate import evaluate_model

        if self.config.training.distributed_backend == "deepspeed":
            logging.warning("Skipping evaluation during DeepSpeed training.")
            return

        if self.config.training.distributed_backend == "fsdp" and self.distributed:
            # Summon full params on rank0 for generation to avoid sharded weight shape issues.
            with FSDP.summon_full_params(
                self.qwen.model, rank0_only=True, writeback=False
            ):
                if self.is_main_process:
                    results = evaluate_model(
                        self.r3,
                        self.val_loader,
                        corruption_levels=[0.0, 0.4, 0.8],
                        max_new_tokens=self.config.evaluation.max_new_tokens,
                        top_k=self.config.retrieval.top_k,
                        return_sums=False,
                    )
                else:
                    results = {}
            dist.barrier()
            if self.is_main_process:
                logging.info("Evaluation:")
                for level, metrics in results.items():
                    logging.info(
                        "  corruption=%.2f | EM %.3f F1 %.3f BLEU %.3f ROUGE-L %.3f",
                        level,
                        metrics["exact_match"],
                        metrics["f1"],
                        metrics["bleu"],
                        metrics["rouge_l"],
                    )
            return

        results = evaluate_model(
            self.r3,
            self.val_loader,
            corruption_levels=[0.0, 0.4, 0.8],
            max_new_tokens=self.config.evaluation.max_new_tokens,
            top_k=self.config.retrieval.top_k,
            return_sums=self.distributed,
        )

        if self.distributed:
            reduced: Dict[float, Dict[str, float]] = {}
            for level, metrics in results.items():
                total = torch.tensor(
                    [
                        metrics.get("exact_match", 0.0),
                        metrics.get("f1", 0.0),
                        metrics.get("bleu", 0.0),
                        metrics.get("rouge_l", 0.0),
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                count = torch.tensor(
                    metrics.get("count", 0.0), device=self.device, dtype=torch.float32
                )
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
                denom = max(count.item(), 1.0)
                reduced[level] = {
                    "exact_match": (total[0] / denom).item(),
                    "f1": (total[1] / denom).item(),
                    "bleu": (total[2] / denom).item(),
                    "rouge_l": (total[3] / denom).item(),
                }
            results = reduced

        if self.is_main_process:
            logging.info("Evaluation:")
            for level, metrics in results.items():
                logging.info(
                    "  corruption=%.2f | EM %.3f F1 %.3f BLEU %.3f ROUGE-L %.3f",
                    level,
                    metrics["exact_match"],
                    metrics["f1"],
                    metrics["bleu"],
                    metrics["rouge_l"],
                )
