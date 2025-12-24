"""Trainer for R3++ multimodal QA."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import functools

from config.train_config import TrainConfig
from data.datasets import UnifiedQACollator, UnifiedQADataset
from models.qwen_wrapper import QwenVLConfig, QwenVLWrapper
from models.r3_modules import R3
from retrieval.image_retrieval import ImageRetriever
from retrieval.text_retrieval import TextRetriever
from training.curriculum import CurriculumScheduler
from training.losses import compute_total_loss


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
    ) -> Any:
        bundle = self.r3.forward_student(
            images,
            questions,
            pseudo_texts,
            answers,
            corruption_level,
            top_k,
        )
        return bundle["outputs"]


class Trainer:
    """Main training loop for R3++."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self._init_distributed()
        self.device = torch.device(
            config.model.device if torch.cuda.is_available() else "cpu"
        )
        self._set_seed(config.training.seed + self.rank)

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
            use_lora=config.model.use_lora,
            lora_r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            lora_target_modules=config.model.lora_target_modules,
        )
        self.qwen = QwenVLWrapper(qwen_cfg)

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

        self.engine = None
        if self.config.training.distributed_backend == "fsdp":
            self._wrap_fsdp()
        elif self.config.training.distributed_backend == "deepspeed":
            self._init_deepspeed()

        self.curriculum = CurriculumScheduler(
            max_corruption=config.curriculum.max_corruption,
            warmup_steps=config.curriculum.warmup_steps,
            total_steps=config.curriculum.total_steps,
        )

        if self.config.training.distributed_backend != "deepspeed":
            self.optimizer = torch.optim.AdamW(
                [p for p in self.qwen.model.parameters() if p.requires_grad]
                + [p for p in self.r3.parameters() if p.requires_grad],
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            self.optimizer = None

        if self.config.training.distributed_backend == "deepspeed":
            self.scaler = None
        else:
            self.scaler = torch.cuda.amp.GradScaler() if config.training.fp16 else None

        self.train_sampler = None
        self.val_sampler = None
        if self.distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
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
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=self.config.training.fsdp_min_num_params,
        )
        self.qwen.model = FSDP(
            self.qwen.model,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            sharding_strategy=sharding,
            use_orig_params=self.config.training.fsdp_use_orig_params,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.device,
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

    def _autocast(self):
        use_amp = self.config.training.fp16 or self.config.training.bf16
        if self.config.training.distributed_backend == "deepspeed" or not use_amp:
            return torch.autocast(device_type=self.device.type, enabled=False)
        dtype = torch.float16 if self.config.training.fp16 else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=True)

    def _save_checkpoint(self, step: int) -> None:
        out_dir = os.path.join(self.config.training.output_dir, f"step_{step}")
        os.makedirs(out_dir, exist_ok=True)
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
        if self.distributed:
            dist.barrier()

    def train(self) -> None:
        total_steps = self.config.training.max_steps
        global_step = 0
        if self.engine is not None:
            self.engine.train()
        else:
            self.qwen.model.train()

        for epoch in range(self.config.training.num_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            for batch in self.train_loader:
                if total_steps is not None and global_step >= total_steps:
                    return

                corruption_level = self.curriculum.get_level(global_step)
                clean = batch["clean"]
                corrupted = batch["corrupted"]

                teacher_outputs = None
                if self.config.training.use_teacher:
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
                        student_outputs = self.engine(
                            corrupted["images"],
                            corrupted["questions"],
                            corrupted["pseudo_texts"],
                            clean["answers"],
                            corruption_level,
                            self.config.retrieval.top_k,
                        )
                        losses = compute_total_loss(
                            student_outputs,
                            teacher_outputs,
                            self.config.loss.consistency_weight,
                            self.config.loss.temperature,
                        )
                        loss = losses["total"]
                    self.engine.backward(loss)
                    self.engine.step()
                else:
                    with self._autocast():
                        student_bundle = self.r3.forward_student(
                            corrupted["images"],
                            corrupted["questions"],
                            corrupted["pseudo_texts"],
                            clean["answers"],
                            corruption_level,
                            self.config.retrieval.top_k,
                        )
                        losses = compute_total_loss(
                            student_bundle["outputs"],
                            teacher_outputs,
                            self.config.loss.consistency_weight,
                            self.config.loss.temperature,
                        )
                        loss = losses["total"] / self.config.training.gradient_accumulation

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (global_step + 1) % self.config.training.gradient_accumulation == 0:
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            assert self.optimizer is not None
                            self.optimizer.step()
                        assert self.optimizer is not None
                        self.optimizer.zero_grad(set_to_none=True)

                if global_step % self.config.training.log_every == 0 and self.is_main_process:
                    logging.info(
                        "step %d | loss %.4f ce %.4f cons %.4f",
                        global_step,
                        losses["total"].item(),
                        losses["ce"].item(),
                        losses["consistency"].item(),
                    )

                if (
                    global_step > 0
                    and global_step % self.config.training.eval_every == 0
                ):
                    self.evaluate()

                if (
                    global_step > 0
                    and global_step % self.config.training.save_every == 0
                ):
                    self._save_checkpoint(global_step)

                global_step += 1

    def evaluate(self) -> None:
        from evaluation.evaluate import evaluate_model

        if self.config.training.distributed_backend == "deepspeed":
            logging.warning("Skipping evaluation during DeepSpeed training.")
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
