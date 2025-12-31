"""Configuration dataclasses for training, retrieval, and evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model and adapter configuration."""
    model_name: str = "models/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "bf16"  # "bf16" or "fp16"
    device: str = "cuda"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    # Default to V/O only for stability on FP16; override via CLI for Q/K ablations.
    lora_target_modules: List[str] = field(default_factory=lambda: ["v_proj", "o_proj"])


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""
    train_jsonl: str = "data/unified/train.jsonl"
    val_jsonl: str = "data/unified/val.jsonl"
    image_root: str = ""
    max_samples: Optional[int] = None
    max_length: int = 4096
    image_size: int = 448
    num_workers: int = 4


@dataclass
class RetrievalConfig:
    """FAISS index and retrieval configuration."""
    index_dir: str = "indices"
    image_index_path: str = "indices/image.index"
    image_meta_path: str = "indices/image.meta.json"
    image_embeds_path: str = "indices/image.embeds.npy"
    text_index_path: str = "indices/text.index"
    text_meta_path: str = "indices/text.meta.json"
    text_embeds_path: str = "indices/text.embeds.npy"
    image_encoder_name: str = "models/clip-vit-b32-laion2B"
    text_encoder_name: str = "models/all-MiniLM-L6-v2"
    top_k: int = 3
    use_retrieval: bool = True


@dataclass
class CorruptionConfig:
    """Corruption probabilities and severity."""
    max_severity: float = 1.6
    blur_prob: float = 0.3
    motion_blur_prob: float = 0.15
    occlusion_prob: float = 0.3
    crop_prob: float = 0.3
    downsample_prob: float = 0.2
    jpeg_prob: float = 0.2
    noise_prob: float = 0.2
    color_prob: float = 0.2
    text_trunc_prob: float = 0.3
    text_noise_prob: float = 0.3
    noise_std: float = 0.1
    jpeg_quality_min: int = 30
    jpeg_quality_max: int = 95
    color_jitter: float = 0.25


@dataclass
class R3Config:
    """R3++ module hyperparameters."""
    prefix_len: int = 8
    latent_tokens: int = 4
    visual_memory_len: int = 4
    hidden_dim: int = 4096
    use_soft_prefix: bool = True
    force_fp32: bool = True
    max_context_chars: int = 0
    enable_corruption: bool = True
    enable_text_retrieval: bool = True
    enable_image_retrieval: bool = True
    enable_prefix: bool = True
    enable_memory: bool = True
    enable_visual_memory: bool = True
    enable_latent_tokens: bool = True
    enable_gate: bool = True
    enable_context: bool = True
    use_score_weighting: bool = True
    score_temperature: float = 1.0
    min_text_score: float = -1.0
    min_image_score: float = -1.0
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)


@dataclass
class LossConfig:
    """Loss weights and temperature."""
    consistency_weight: float = 0.5
    temperature: float = 1.0
    gate_conf_weight: float = 0.1
    gate_entropy_weight: float = 0.01
    retrieval_align_weight: float = 0.05
    retrieval_align_temperature: float = 0.07


@dataclass
class CurriculumConfig:
    """Corruption curriculum schedule."""
    max_corruption: float = 0.8
    warmup_steps: int = 1000
    total_steps: int = 10000


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    output_dir: str = "checkpoints"
    batch_size: int = 1
    num_epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    adam_eps: float = 1e-6
    max_grad_norm: float = 1.0
    min_label_ratio: float = 0.0
    gradient_accumulation: int = 1
    max_steps: Optional[int] = None
    warmup_steps: int = 0
    eval_every: int = 1000
    save_every: int = 1000
    log_every: int = 50
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    loss_scale: float = 256.0
    disable_grad_scaler: bool = False
    skip_nonfinite_grads: bool = True
    lora_lr_mult: float = 0.1
    r3_lr_mult: float = 1.0
    sampling_alpha: Optional[float] = None
    sample_every: int = 0
    sample_num: int = 1
    sample_max_new_tokens: int = 32
    gradient_checkpointing: bool = False
    distributed_backend: str = "none"  # "none", "fsdp", "deepspeed"
    deepspeed_config: Optional[str] = None
    fsdp_min_num_params: int = 100_000_000
    fsdp_cpu_offload: bool = False
    fsdp_use_orig_params: bool = True
    fsdp_sharding: str = "full"  # "full", "grad", "no_shard"
    use_teacher: bool = True
    teacher_mode: str = "ema"  # "copy", "ema", "shared"
    teacher_ema_decay: float = 0.999
    teacher_ema_update_steps: int = 1
    teacher_ema_start_step: int = 0
    resume_from: Optional[str] = None
    resume_optimizer: bool = False
    resume_r3: bool = True
    resume_lora: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    batch_size: int = 1
    max_new_tokens: int = 64
    max_eval_samples: Optional[int] = 200


@dataclass
class TrainConfig:
    """Top-level configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    r3: R3Config = field(default_factory=R3Config)
    loss: LossConfig = field(default_factory=LossConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        """Convert the configuration to a serializable dict."""
        return asdict(self)
