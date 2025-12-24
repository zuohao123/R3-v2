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
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""
    train_jsonl: str = "data/unified/train.jsonl"
    val_jsonl: str = "data/unified/val.jsonl"
    image_root: str = ""
    max_samples: Optional[int] = None
    max_length: int = 256
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
    image_encoder_name: str = "models/clip-vit-base-patch32"
    text_encoder_name: str = "models/all-MiniLM-L6-v2"
    top_k: int = 5
    use_retrieval: bool = True


@dataclass
class CorruptionConfig:
    """Corruption probabilities and severity."""
    max_severity: float = 0.8
    blur_prob: float = 0.3
    occlusion_prob: float = 0.3
    crop_prob: float = 0.3
    text_trunc_prob: float = 0.3
    text_noise_prob: float = 0.3


@dataclass
class R3Config:
    """R3++ module hyperparameters."""
    prefix_len: int = 8
    latent_tokens: int = 4
    hidden_dim: int = 4096
    use_soft_prefix: bool = False
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)


@dataclass
class LossConfig:
    """Loss weights and temperature."""
    consistency_weight: float = 0.5
    temperature: float = 1.0


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
    gradient_accumulation: int = 1
    max_steps: Optional[int] = None
    warmup_steps: int = 0
    eval_every: int = 1000
    save_every: int = 1000
    log_every: int = 50
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    distributed_backend: str = "none"  # "none", "fsdp", "deepspeed"
    deepspeed_config: Optional[str] = None
    fsdp_min_num_params: int = 100_000_000
    fsdp_cpu_offload: bool = False
    fsdp_use_orig_params: bool = True
    fsdp_sharding: str = "full"  # "full", "grad", "no_shard"
    use_teacher: bool = True


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
