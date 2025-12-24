"""Wrapper for Qwen3-VL-8B with optional LoRA and teacher model."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


@dataclass
class QwenVLConfig:
    """Configuration for Qwen VL wrapper."""
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "bf16"
    device: str = "cuda"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None


class QwenVLWrapper:
    """Thin wrapper around Qwen3-VL for training and generation."""

    def __init__(self, config: QwenVLConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
        dtype = self._resolve_dtype(config.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.train()

        if config.use_lora:
            self._apply_lora()

        self.teacher = copy.deepcopy(self.model)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def _apply_lora(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("peft is required for LoRA training.") from exc
        target_modules = self.config.lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
        lora_cfg = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()

    @staticmethod
    def _resolve_dtype(dtype_str: str) -> torch.dtype:
        if dtype_str == "fp16":
            return torch.float16
        if dtype_str == "bf16":
            return torch.bfloat16
        return torch.float32

    def build_prompt(self, question: str, pseudo_text: Optional[str] = None) -> str:
        """Build a prompt with image placeholder and optional pseudo-text."""
        # TODO: Swap to the official Qwen3-VL chat template if needed.
        context = f"Context: {pseudo_text}\n" if pseudo_text else ""
        return f"<image>\nQuestion: {question}\n{context}Answer:"

    def _build_labels(
        self, prompt_texts: List[str], full_texts: List[str], max_length: Optional[int]
    ) -> torch.Tensor:
        tokenizer = self.processor.tokenizer
        prompt_ids = tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"]
        full_ids = tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"]
        labels = full_ids.clone()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for i in range(labels.size(0)):
            prompt_len = (prompt_ids[i] != pad_id).sum().item()
            labels[i, :prompt_len] = -100
        return labels

    def encode_inputs(
        self,
        images: List[Any],
        questions: List[str],
        pseudo_texts: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        prompts = [
            self.build_prompt(q, p)
            for q, p in zip(questions, pseudo_texts or [""] * len(questions))
        ]
        if answers is not None:
            full_texts = [f"{p} {a}" for p, a in zip(prompts, answers)]
        else:
            full_texts = prompts
        inputs = self.processor(
            images=images,
            text=full_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if answers is not None:
            labels = self._build_labels(prompts, full_texts, max_length=max_length)
            inputs["labels"] = labels
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def forward_student(
        self,
        images: List[Any],
        questions: List[str],
        pseudo_texts: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        prefix_embeds: Optional[torch.Tensor] = None,
        use_soft_prefix: bool = False,
    ) -> Any:
        inputs = self.encode_inputs(images, questions, pseudo_texts, answers, max_length)
        if use_soft_prefix and prefix_embeds is not None:
            input_ids = inputs.pop("input_ids")
            attention_mask = inputs.pop("attention_mask")
            token_embeds = self.model.get_input_embeddings()(input_ids)
            prefix_embeds = prefix_embeds.to(token_embeds.dtype).to(token_embeds.device)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            prefix_mask = torch.ones(
                (attention_mask.size(0), prefix_embeds.size(1)),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            labels = inputs.get("labels")
            if labels is not None:
                prefix_ignore = torch.full(
                    (labels.size(0), prefix_embeds.size(1)),
                    -100,
                    device=labels.device,
                    dtype=labels.dtype,
                )
                labels = torch.cat([prefix_ignore, labels], dim=1)
                inputs["labels"] = labels
            return self.model(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, **inputs
            )
        return self.model(**inputs)

    def forward_teacher(
        self,
        images: List[Any],
        questions: List[str],
        pseudo_texts: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        max_length: Optional[int] = None,
    ) -> Any:
        inputs = self.encode_inputs(images, questions, pseudo_texts, answers, max_length)
        with torch.no_grad():
            return self.teacher(**inputs)

    def generate_answer(
        self,
        images: List[Any],
        questions: List[str],
        pseudo_texts: Optional[List[str]] = None,
        max_new_tokens: int = 64,
    ) -> List[str]:
        prompts = [
            self.build_prompt(q, p)
            for q, p in zip(questions, pseudo_texts or [""] * len(questions))
        ]
        inputs = self.processor(
            images=images,
            text=prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        decoded = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers = []
        for text in decoded:
            if "Answer:" in text:
                answers.append(text.split("Answer:", 1)[-1].strip())
            else:
                answers.append(text.strip())
        return answers
