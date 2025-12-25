"""Trainer for R3++ multimodal QA."""
from __future__ import annotations

import itertools
import json
import logging
import math
import os
import time
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
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
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
        return bundle["outputs"]


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
            if config.training.fp16 and self.distributed:
                self.scaler = ShardedGradScaler()
            elif config.training.fp16:
                self.scaler = torch.cuda.amp.GradScaler()
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
        if self.config.training.fsdp_min_num_params > 0:
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

    def _clip_gradients(self) -> None:
        max_norm = self.config.training.max_grad_norm
        if not max_norm or max_norm <= 0:
            return
        if self.config.training.distributed_backend == "fsdp":
            FSDP.clip_grad_norm_(self.qwen.model, max_norm)
            torch.nn.utils.clip_grad_norm_(self.r3.parameters(), max_norm)
            return
        params = [
            p
            for p in itertools.chain(self.qwen.model.parameters(), self.r3.parameters())
            if p.requires_grad
        ]
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm)

    @staticmethod
    def _truncate(text: str, max_len: int = 240) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _log_samples(self, batch: Dict[str, Any], corruption_level: float, step: int) -> None:
        sample_every = self.config.training.sample_every
        if sample_every <= 0 or not self.is_main_process:
            return
        if self.config.training.distributed_backend in {"deepspeed", "fsdp"}:
            if step == 0 and self.is_main_process:
                logging.warning("Sample logging is disabled under %s.", self.config.training.distributed_backend)
            return
        if step % sample_every != 0:
            return

        clean = batch["clean"]
        corrupted = batch["corrupted"]
        n = min(self.config.training.sample_num, len(clean["questions"]))
        if n <= 0:
            return
        images = corrupted["images"][:n]
        questions = corrupted["questions"][:n]
        pseudo_texts = corrupted["pseudo_texts"][:n]
        answers = clean["answers"][:n]

        self.qwen.model.eval()
        self.r3.eval()
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
        preds, retrieved_texts, _, contexts = outputs
        self.qwen.model.train()
        self.r3.train()

        logging.info("Sample outputs at step %d (corruption=%.2f):", step, corruption_level)
        for idx in range(n):
            logging.info("  Q: %s", self._truncate(questions[idx]))
            logging.info("  GT: %s", self._truncate(answers[idx]))
            logging.info("  Pred: %s", self._truncate(preds[idx]))
            if contexts[idx]:
                logging.info("  Context: %s", self._truncate(contexts[idx]))
            if retrieved_texts[idx]:
                shown = retrieved_texts[idx][: min(3, len(retrieved_texts[idx]))]
                joined = " || ".join(self._truncate(t) for t in shown if t)
                if joined:
                    logging.info("  Retrieved: %s", joined)

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
                            self.config.data.max_length,
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
                            max_length=self.config.data.max_length,
                        )
                        losses = compute_total_loss(
                            student_bundle["outputs"],
                            teacher_outputs,
                            self.config.loss.consistency_weight,
                            self.config.loss.temperature,
                        )
                        loss = losses["total"] / self.config.training.gradient_accumulation

                    label_ratio_val = None
                    if "label_ratio" in losses:
                        label_ratio_val = float(losses["label_ratio"].item())
                    if (
                        self.config.training.min_label_ratio > 0
                        and label_ratio_val is not None
                        and label_ratio_val < self.config.training.min_label_ratio
                    ):
                        if self.is_main_process:
                            logging.warning(
                                "Low label ratio %.4f at step %d; skipping update.",
                                label_ratio_val,
                                global_step,
                            )
                        if self.optimizer is not None:
                            self.optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    if not torch.isfinite(losses["total"]):
                        if self.is_main_process:
                            logging.warning("Non-finite loss at step %d; skipping update.", global_step)
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
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            self._clip_gradients()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            assert self.optimizer is not None
                            self._clip_gradients()
                            self.optimizer.step()
                        assert self.optimizer is not None
                        self.optimizer.zero_grad(set_to_none=True)

                if global_step % self.config.training.log_every == 0 and self.is_main_process:
                    lr = None
                    if self.optimizer is not None and self.optimizer.param_groups:
                        lr = self.optimizer.param_groups[0].get("lr")
                    scale = self.scaler.get_scale() if self.scaler is not None else None
                    gate_stats = None
                    conf_stats = None
                    if self.config.training.distributed_backend != "deepspeed":
                        if isinstance(student_bundle, dict) and "gates" in student_bundle:
                            gates = student_bundle["gates"].mean(dim=0).tolist()
                            gate_stats = tuple(round(g, 3) for g in gates)
                        if isinstance(student_bundle, dict) and "c_vis" in student_bundle:
                            c_vis = student_bundle["c_vis"].mean().item()
                            c_text = student_bundle["c_text"].mean().item()
                            conf_stats = (round(c_vis, 3), round(c_text, 3))
                    elapsed = time.time() - start_time
                    eta = None
                    if total_steps is not None and global_step > 0:
                        rate = elapsed / global_step
                        eta = rate * max(total_steps - global_step, 0)
                    label_ratio = None
                    if "label_ratio" in losses:
                        label_ratio = float(losses["label_ratio"].item())
                    logging.info(
                        "epoch %d step %d | loss %.4f ce %.4f cons %.4f | corr %.2f | lr %s | scale %s | gate %s | conf %s | label %.3f | eta %s",
                        epoch,
                        global_step,
                        losses["total"].item(),
                        losses["ce"].item(),
                        losses["consistency"].item(),
                        corruption_level,
                        f"{lr:.2e}" if lr is not None else "n/a",
                        f"{scale:.1f}" if scale is not None else "n/a",
                        gate_stats if gate_stats is not None else "n/a",
                        conf_stats if conf_stats is not None else "n/a",
                        label_ratio if label_ratio is not None else -1.0,
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

        if pbar is not None:
            pbar.close()

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
