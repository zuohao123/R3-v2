"""R3++ modules for corruption, retrieval fusion, and reconstruction."""
from __future__ import annotations

import contextlib
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter
import numpy as np
import torch
from torch import nn

from config.train_config import R3Config
from models.qwen_wrapper import QwenVLWrapper
from retrieval.image_retrieval import ImageRetriever
from retrieval.text_retrieval import TextRetriever


class CorruptionSimulator(nn.Module):
    """Simulate partial modality corruption on images and text."""

    def __init__(self, config: R3Config) -> None:
        super().__init__()
        self.config = config

    def _blur(self, image: Image.Image, severity: float) -> Image.Image:
        radius = max(0.1, 2.0 * severity)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _occlude(self, image: Image.Image, severity: float) -> Image.Image:
        width, height = image.size
        occlusion_ratio = 0.2 + 0.3 * severity
        occ_w = int(width * occlusion_ratio)
        occ_h = int(height * occlusion_ratio)
        x0 = random.randint(0, max(0, width - occ_w))
        y0 = random.randint(0, max(0, height - occ_h))
        overlay = Image.new("RGB", (occ_w, occ_h), color=(0, 0, 0))
        image = image.copy()
        image.paste(overlay, (x0, y0))
        return image

    def _crop(self, image: Image.Image, severity: float) -> Image.Image:
        width, height = image.size
        crop_ratio = 0.8 - 0.3 * severity
        crop_w = int(width * crop_ratio)
        crop_h = int(height * crop_ratio)
        x0 = random.randint(0, max(0, width - crop_w))
        y0 = random.randint(0, max(0, height - crop_h))
        cropped = image.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        return cropped.resize((width, height))

    def _truncate_text(self, text: str, severity: float) -> str:
        if not text:
            return text
        keep_ratio = max(0.1, 1.0 - 0.5 * severity)
        keep_len = max(1, int(len(text) * keep_ratio))
        return text[:keep_len]

    def _add_text_noise(self, text: str, severity: float) -> str:
        if not text:
            return text
        chars = list(text)
        num_noisy = max(1, int(len(chars) * 0.1 * severity))
        for _ in range(num_noisy):
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
        return "".join(chars)

    def forward(
        self, images: List[Image.Image], texts: List[str], level: float
    ) -> Tuple[List[Image.Image], List[str], torch.Tensor, torch.Tensor]:
        cfg = self.config.corruption
        corrupted_images: List[Image.Image] = []
        corrupted_texts: List[str] = []
        vis_conf: List[float] = []
        text_conf: List[float] = []

        for image, text in zip(images, texts):
            vis_severity = 0.0
            txt_severity = 0.0
            if random.random() < cfg.blur_prob * level:
                image = self._blur(image, level)
                vis_severity += 0.3
            if random.random() < cfg.occlusion_prob * level:
                image = self._occlude(image, level)
                vis_severity += 0.4
            if random.random() < cfg.crop_prob * level:
                image = self._crop(image, level)
                vis_severity += 0.3

            if random.random() < cfg.text_trunc_prob * level:
                text = self._truncate_text(text, level)
                txt_severity += 0.4
            if random.random() < cfg.text_noise_prob * level:
                text = self._add_text_noise(text, level)
                txt_severity += 0.3

            corrupted_images.append(image)
            corrupted_texts.append(text)
            vis_conf.append(max(0.0, 1.0 - vis_severity))
            text_conf.append(max(0.0, 1.0 - txt_severity))

        c_vis = torch.tensor(vis_conf).unsqueeze(-1)
        c_text = torch.tensor(text_conf).unsqueeze(-1)
        return corrupted_images, corrupted_texts, c_vis, c_text


class PrefixEnhancer(nn.Module):
    """Summarize retrieved text embeddings into prefix embeddings."""

    def __init__(self, text_dim: int, hidden_dim: int, prefix_len: int) -> None:
        super().__init__()
        self.prefix_len = prefix_len
        self.proj = nn.Linear(text_dim, hidden_dim)
        self.to_prefix = nn.Linear(hidden_dim, hidden_dim * prefix_len)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, text_embeds: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if weights is None:
            pooled = text_embeds.mean(dim=1)
        else:
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled = (text_embeds * weights.unsqueeze(-1)).sum(dim=1)
        hidden = torch.tanh(self.proj(pooled))
        prefix = self.to_prefix(hidden)
        prefix = prefix.view(prefix.size(0), self.prefix_len, -1)
        return self.norm(prefix)


class MemoryAligner(nn.Module):
    """Align retrieved text and image embeddings to shared memory space."""

    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

    def forward(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        text_weights: Optional[torch.Tensor] = None,
        image_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if text_weights is None:
            pooled_text = text_embeds.mean(dim=1)
        else:
            text_weights = text_weights / text_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled_text = (text_embeds * text_weights.unsqueeze(-1)).sum(dim=1)
        if image_weights is None:
            pooled_image = image_embeds.mean(dim=1)
        else:
            image_weights = image_weights / image_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled_image = (image_embeds * image_weights.unsqueeze(-1)).sum(dim=1)
        mem_t = torch.tanh(self.text_proj(pooled_text))
        mem_i = torch.tanh(self.image_proj(pooled_image))
        return mem_t, mem_i


class VisualMemoryTokens(nn.Module):
    """Project retrieved image embeddings into memory tokens."""

    def __init__(self, image_dim: int, hidden_dim: int, memory_len: int) -> None:
        super().__init__()
        self.memory_len = memory_len
        self.proj = nn.Linear(image_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, image_embeds: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, top_k, dim = image_embeds.shape
        if weights is None:
            weights = torch.full(
                (batch, top_k), 1.0 / top_k, device=image_embeds.device
            )
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        take = min(self.memory_len, top_k)
        _, indices = torch.topk(weights, k=take, dim=1)
        gather = indices.unsqueeze(-1).expand(-1, -1, dim)
        selected = torch.gather(image_embeds, 1, gather)
        selected_weights = torch.gather(weights, 1, indices)
        tokens = self.proj(selected) * selected_weights.unsqueeze(-1)
        if self.memory_len > take:
            pad = torch.zeros(
                (batch, self.memory_len - take, tokens.size(-1)),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = torch.cat([tokens, pad], dim=1)
        return self.norm(tokens)


class LatentImputationTokens(nn.Module):
    """Learnable imputation tokens conditioned on confidence."""

    def __init__(self, num_tokens: int, hidden_dim: int) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)

    def forward(self, batch_size: int, confidence: torch.Tensor) -> torch.Tensor:
        confidence = confidence.clamp(0.0, 1.0)
        scale = (1.0 - confidence).unsqueeze(-1)
        tokens = self.tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return tokens * scale


class AdaptiveGate(nn.Module):
    """Adaptive gating across text memory, image memory, and visual features."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self, mem_t: torch.Tensor, mem_i: torch.Tensor, vis_feat: torch.Tensor
    ) -> torch.Tensor:
        fused = torch.cat([mem_t, mem_i, vis_feat], dim=-1)
        logits = self.fc(fused)
        return torch.softmax(logits, dim=-1)


class R3(nn.Module):
    """Orchestrate corruption, retrieval, and reconstruction."""

    def __init__(
        self,
        qwen: QwenVLWrapper,
        text_retriever: Optional[TextRetriever],
        image_retriever: Optional[ImageRetriever],
        config: R3Config,
    ) -> None:
        super().__init__()
        self.qwen = qwen
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.config = config
        self.corruptor = CorruptionSimulator(config)

        text_dim = 768
        image_dim = 512
        if text_retriever is not None and text_retriever.store is not None:
            text_dim = text_retriever.store.dim
        if image_retriever is not None and image_retriever.store is not None:
            image_dim = image_retriever.store.dim
        self.text_dim = text_dim
        self.image_dim = image_dim

        self.prefix_enhancer = (
            PrefixEnhancer(text_dim, config.hidden_dim, config.prefix_len)
            if config.enable_prefix
            else None
        )
        self.memory_aligner = (
            MemoryAligner(text_dim, image_dim, config.hidden_dim)
            if config.enable_memory
            else None
        )
        self.visual_memory = (
            VisualMemoryTokens(image_dim, config.hidden_dim, config.visual_memory_len)
            if config.enable_visual_memory
            else None
        )
        self.latent_tokens = (
            LatentImputationTokens(config.latent_tokens, config.hidden_dim)
            if config.enable_latent_tokens
            else None
        )
        self.gate = AdaptiveGate(config.hidden_dim) if config.enable_gate else None
        self.vis_proj = nn.Linear(image_dim, config.hidden_dim)

    @staticmethod
    def _sanitize(tensor: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        return torch.nan_to_num(tensor, nan=fill, posinf=fill, neginf=fill)

    def _compose_context(
        self,
        retrieved_texts: List[List[str]],
        retrieved_images: List[List[str]],
        gates: torch.Tensor,
        top_k: int,
        text_scores: Optional[torch.Tensor] = None,
        image_scores: Optional[torch.Tensor] = None,
    ) -> List[str]:
        contexts = []
        for idx, (texts, imgs) in enumerate(zip(retrieved_texts, retrieved_images)):
            g_t, g_i, g_v = gates[idx].tolist()
            if text_scores is not None:
                scores = text_scores[idx].tolist()
                texts = [
                    text
                    for text, score in zip(texts, scores)
                    if score >= self.config.min_text_score
                ]
            if image_scores is not None:
                scores = image_scores[idx].tolist()
                imgs = [
                    text
                    for text, score in zip(imgs, scores)
                    if score >= self.config.min_image_score
                ]
            n_text = max(1, int(round(g_t * top_k))) if texts else 0
            n_img = max(0, int(round(g_i * top_k))) if imgs else 0
            selected = []
            selected.extend(texts[:n_text])
            selected.extend(imgs[:n_img])
            context = " ".join(selected)
            if context and g_v > 0.5:
                context += " [VISUAL_CONF_HIGH]"
            contexts.append(context.strip())
        return contexts

    def _score_weights(
        self, scores: Optional[np.ndarray], top_k: int, min_score: float
    ) -> torch.Tensor:
        if scores is None:
            return torch.full((top_k,), 1.0 / top_k, device=self.qwen.device)
        weights = torch.tensor(scores, device=self.qwen.device, dtype=torch.float32)
        if min_score > -1.0:
            weights = weights.masked_fill(weights < min_score, float("-inf"))
        if self.config.use_score_weighting:
            temperature = max(self.config.score_temperature, 1e-6)
            weights = torch.softmax(weights / temperature, dim=-1)
        else:
            mask = torch.isfinite(weights)
            weights = mask.float()
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = weights.sum(dim=-1, keepdim=True)
        fallback = row_sum < 1e-6
        if fallback.any():
            weights = torch.where(
                fallback,
                torch.full_like(weights, 1.0 / weights.size(-1)),
                weights / row_sum.clamp_min(1e-6),
            )
        else:
            weights = weights / row_sum.clamp_min(1e-6)
        return weights

    def _retrieve_texts(
        self, queries: List[str], top_k: int
    ) -> Tuple[torch.Tensor, List[List[str]], Optional[np.ndarray]]:
        if self.text_retriever is None or not self.config.enable_text_retrieval:
            dummy = torch.zeros((len(queries), top_k, self.text_dim))
            return dummy, [[] for _ in queries], None
        result = self.text_retriever.retrieve(queries, top_k)
        embeds = result.get("embeddings")
        if embeds is None:
            texts = [[meta.get("pseudo_text", "") for meta in row] for row in result["metadata"]]
            embeds = self.text_retriever.encode_texts([t for row in texts for t in row])
            embeds = embeds.reshape(len(queries), top_k, -1)
        texts = [[meta.get("pseudo_text", "") for meta in row] for row in result["metadata"]]
        return torch.from_numpy(embeds).float(), texts, result.get("scores")

    def _retrieve_images(
        self, images: List[Image.Image], top_k: int
    ) -> Tuple[torch.Tensor, List[List[str]], List[List[str]], Optional[np.ndarray]]:
        if self.image_retriever is None or not self.config.enable_image_retrieval:
            dummy = torch.zeros((len(images), top_k, self.image_dim))
            return dummy, [[] for _ in images], [[] for _ in images], None
        result = self.image_retriever.retrieve(images, top_k)
        embeds = result.get("embeddings")
        if embeds is None:
            image_paths = [
                meta.get("image_path", "") for row in result["metadata"] for meta in row
            ]
            pil_images = []
            for path in image_paths:
                try:
                    pil_images.append(Image.open(path).convert("RGB"))
                except (FileNotFoundError, OSError):
                    pil_images.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
            embeds = self.image_retriever.encode_images(pil_images)
            embeds = embeds.reshape(len(images), top_k, -1)
        texts = [
            [meta.get("pseudo_text", "") for meta in row] for row in result["metadata"]
        ]
        paths = [
            [meta.get("image_path", "") for meta in row] for row in result["metadata"]
        ]
        return torch.from_numpy(embeds).float(), texts, paths, result.get("scores")

    def forward_student(
        self,
        images: List[Image.Image],
        questions: List[str],
        pseudo_texts: List[str],
        answers: Optional[List[str]],
        corruption_level: float,
        top_k: int,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        device_type = self.qwen.device.type
        amp_ctx = contextlib.nullcontext()
        if self.config.force_fp32:
            amp_ctx = torch.autocast(device_type=device_type, enabled=False)

        with amp_ctx:
            if self.config.enable_corruption:
                corr_images, corr_texts, c_vis, c_text = self.corruptor(
                    images, pseudo_texts, corruption_level
                )
            else:
                corr_images = images
                corr_texts = pseudo_texts
                c_vis = torch.ones(len(images), 1)
                c_text = torch.ones(len(images), 1)

            queries = [f"{q} {t}" for q, t in zip(questions, corr_texts)]
            text_embeds, retrieved_texts, text_scores = self._retrieve_texts(queries, top_k)
            image_embeds, retrieved_images, retrieved_image_paths, image_scores = self._retrieve_images(
                corr_images, top_k
            )

            text_embeds = text_embeds.to(self.qwen.device, dtype=torch.float32)
            image_embeds = image_embeds.to(self.qwen.device, dtype=torch.float32)
            c_vis = c_vis.to(self.qwen.device, dtype=torch.float32)
            c_text = c_text.to(self.qwen.device, dtype=torch.float32)

            text_embeds = self._sanitize(text_embeds)
            image_embeds = self._sanitize(image_embeds)
            text_weights = (
                self._score_weights(text_scores, top_k, self.config.min_text_score)
                if text_scores is not None
                else torch.full((len(images), top_k), 1.0 / top_k, device=self.qwen.device)
            )
            image_weights = (
                self._score_weights(image_scores, top_k, self.config.min_image_score)
                if image_scores is not None
                else torch.full((len(images), top_k), 1.0 / top_k, device=self.qwen.device)
            )

            prefix_tokens: List[torch.Tensor] = []
            if self.prefix_enhancer is not None:
                text_prefix = self.prefix_enhancer(text_embeds, text_weights)
                prefix_tokens.append(text_prefix)
            if self.visual_memory is not None:
                visual_prefix = self.visual_memory(image_embeds, image_weights)
                prefix_tokens.append(visual_prefix)
            if self.latent_tokens is not None:
                prefix_tokens.append(self.latent_tokens(len(images), (c_vis + c_text) / 2))

            retrieval_ready = (
                self.memory_aligner is not None
                and self.config.enable_text_retrieval
                and self.config.enable_image_retrieval
            )
            if self.memory_aligner is not None:
                mem_t, mem_i = self.memory_aligner(
                    text_embeds, image_embeds, text_weights=text_weights, image_weights=image_weights
                )
                mem_t = mem_t * c_text
                mem_i = mem_i * c_vis
            else:
                mem_t = torch.zeros(
                    (len(images), self.config.hidden_dim),
                    device=self.qwen.device,
                    dtype=torch.float32,
                )
                mem_i = torch.zeros(
                    (len(images), self.config.hidden_dim),
                    device=self.qwen.device,
                    dtype=torch.float32,
                )

            mem_t = self._sanitize(mem_t)
            mem_i = self._sanitize(mem_i)
            vis_feat = self.vis_proj(image_embeds.mean(dim=1))
            vis_feat = self._sanitize(vis_feat)
            if self.gate is not None:
                gates = self.gate(mem_t, mem_i, vis_feat)
                gates = self._sanitize(gates, fill=1.0 / 3)
                gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            else:
                gates = torch.full((len(images), 3), 1 / 3, device=self.qwen.device)

            prefix = None
            if prefix_tokens:
                scaled_tokens: List[torch.Tensor] = []
                token_idx = 0
                if self.prefix_enhancer is not None:
                    scaled_tokens.append(prefix_tokens[token_idx] * gates[:, 0:1].unsqueeze(-1))
                    token_idx += 1
                if self.visual_memory is not None:
                    scaled_tokens.append(prefix_tokens[token_idx] * gates[:, 1:2].unsqueeze(-1))
                    token_idx += 1
                if self.latent_tokens is not None:
                    scaled_tokens.append(prefix_tokens[token_idx])
                    token_idx += 1
                prefix = torch.cat(scaled_tokens, dim=1)
                prefix = self._sanitize(prefix)

            if self.config.enable_context:
                contexts = self._compose_context(
                    retrieved_texts,
                    retrieved_images,
                    gates,
                    top_k,
                    text_scores=text_scores,
                    image_scores=image_scores,
                )
                aug_pseudo_texts = [
                    f"{text} {ctx}".strip() if ctx else text
                    for text, ctx in zip(corr_texts, contexts)
                ]
            else:
                aug_pseudo_texts = corr_texts

            if self.config.max_context_chars > 0:
                aug_pseudo_texts = [
                    text[: self.config.max_context_chars] for text in aug_pseudo_texts
                ]

        outputs = self.qwen.forward_student(
            corr_images,
            questions,
            aug_pseudo_texts,
            answers,
            max_length=max_length,
            prefix_embeds=prefix,
            use_soft_prefix=self.config.use_soft_prefix and prefix is not None,
        )

        return {
            "outputs": outputs,
            "gates": gates.detach(),
            "c_vis": c_vis.detach(),
            "c_text": c_text.detach(),
            "retrieved_texts": retrieved_texts,
            "retrieved_image_paths": retrieved_image_paths,
            "mem_t": mem_t,
            "mem_i": mem_i,
            "retrieval_ready": retrieval_ready,
        }

    def generate(
        self,
        images: List[Image.Image],
        questions: List[str],
        pseudo_texts: List[str],
        corruption_level: float,
        top_k: int,
        max_new_tokens: int,
        return_retrieval: bool = False,
    ) -> Any:
        if self.config.enable_corruption:
            corr_images, corr_texts, _, _ = self.corruptor(
                images, pseudo_texts, corruption_level
            )
        else:
            corr_images = images
            corr_texts = pseudo_texts
        queries = [f"{q} {t}" for q, t in zip(questions, corr_texts)]
        text_embeds, retrieved_texts, text_scores = self._retrieve_texts(queries, top_k)
        image_embeds, retrieved_images, retrieved_image_paths, image_scores = self._retrieve_images(
            corr_images, top_k
        )
        text_embeds = text_embeds.to(self.qwen.device, dtype=torch.float32)
        image_embeds = image_embeds.to(self.qwen.device, dtype=torch.float32)
        text_embeds = self._sanitize(text_embeds)
        image_embeds = self._sanitize(image_embeds)
        text_weights = (
            self._score_weights(text_scores, top_k, self.config.min_text_score)
            if text_scores is not None
            else torch.full((len(images), top_k), 1.0 / top_k, device=self.qwen.device)
        )
        image_weights = (
            self._score_weights(image_scores, top_k, self.config.min_image_score)
            if image_scores is not None
            else torch.full((len(images), top_k), 1.0 / top_k, device=self.qwen.device)
        )
        if self.memory_aligner is not None:
            mem_t, mem_i = self.memory_aligner(
                text_embeds, image_embeds, text_weights=text_weights, image_weights=image_weights
            )
            mem_t = self._sanitize(mem_t)
            mem_i = self._sanitize(mem_i)
        else:
            mem_t = torch.zeros(
                (len(images), self.config.hidden_dim),
                device=self.qwen.device,
                dtype=torch.float32,
            )
            mem_i = torch.zeros(
                (len(images), self.config.hidden_dim),
                device=self.qwen.device,
                dtype=torch.float32,
            )
        vis_feat = self.vis_proj(image_embeds.mean(dim=1))
        vis_feat = self._sanitize(vis_feat)
        if self.gate is not None:
            gates = self.gate(mem_t, mem_i, vis_feat)
            gates = self._sanitize(gates, fill=1.0 / 3)
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            gates = torch.full((len(images), 3), 1 / 3, device=self.qwen.device)
        prefix = None
        if self.config.use_soft_prefix:
            prefix_tokens: List[torch.Tensor] = []
            if self.prefix_enhancer is not None:
                prefix_tokens.append(self.prefix_enhancer(text_embeds, text_weights))
            if self.visual_memory is not None:
                prefix_tokens.append(self.visual_memory(image_embeds, image_weights))
            if self.latent_tokens is not None:
                prefix_tokens.append(
                    self.latent_tokens(len(images), torch.ones(len(images), 1, device=self.qwen.device))
                )
            if prefix_tokens:
                scaled_tokens: List[torch.Tensor] = []
                token_idx = 0
                if self.prefix_enhancer is not None:
                    scaled_tokens.append(prefix_tokens[token_idx] * gates[:, 0:1].unsqueeze(-1))
                    token_idx += 1
                if self.visual_memory is not None:
                    scaled_tokens.append(prefix_tokens[token_idx] * gates[:, 1:2].unsqueeze(-1))
                    token_idx += 1
                if self.latent_tokens is not None:
                    scaled_tokens.append(prefix_tokens[token_idx])
                prefix = self._sanitize(torch.cat(scaled_tokens, dim=1))
        if self.config.enable_context:
            contexts = self._compose_context(
                retrieved_texts,
                retrieved_images,
                gates,
                top_k,
                text_scores=text_scores,
                image_scores=image_scores,
            )
            aug_pseudo_texts = [
                f"{text} {ctx}".strip() if ctx else text
                for text, ctx in zip(corr_texts, contexts)
            ]
        else:
            contexts = ["" for _ in corr_texts]
            aug_pseudo_texts = corr_texts
        if self.config.max_context_chars > 0:
            aug_pseudo_texts = [
                text[: self.config.max_context_chars] for text in aug_pseudo_texts
            ]
        if return_retrieval:
            answers, prompts = self.qwen.generate_answer(
                corr_images,
                questions,
                aug_pseudo_texts,
                max_new_tokens=max_new_tokens,
                return_prompts=True,
                prefix_embeds=prefix,
                use_soft_prefix=self.config.use_soft_prefix and prefix is not None,
            )
            return answers, retrieved_texts, retrieved_image_paths, contexts, prompts
        answers = self.qwen.generate_answer(
            corr_images,
            questions,
            aug_pseudo_texts,
            max_new_tokens=max_new_tokens,
            prefix_embeds=prefix,
            use_soft_prefix=self.config.use_soft_prefix and prefix is not None,
        )
        return answers
