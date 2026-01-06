"""R3++ modules for corruption, retrieval fusion, and reconstruction."""
from __future__ import annotations

import contextlib
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torch
import torch.nn.functional as F
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

    def _motion_blur(self, image: Image.Image, severity: float) -> Image.Image:
        # Use numpy shift-average to avoid Pillow kernel size restrictions.
        arr = np.asarray(image).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        k = 3 + 2 * int(round(severity * 2))
        k = max(3, min(7, k))
        axis = 1 if random.random() < 0.5 else 0
        passes = 2 if severity >= 0.8 else 1
        for _ in range(passes):
            acc = np.zeros_like(arr)
            offsets = range(-(k // 2), k // 2 + 1)
            for off in offsets:
                acc += np.roll(arr, off, axis=axis)
            arr = acc / float(len(list(offsets)))
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return Image.fromarray(arr)

    def _downsample(self, image: Image.Image, severity: float) -> Image.Image:
        width, height = image.size
        scale = max(0.3, 1.0 - 0.6 * severity)
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        small = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return small.resize((width, height), resample=Image.BILINEAR)

    def _jpeg_compress(self, image: Image.Image, severity: float) -> Image.Image:
        quality = int(
            self.config.corruption.jpeg_quality_max
            - severity
            * (
                self.config.corruption.jpeg_quality_max
                - self.config.corruption.jpeg_quality_min
            )
        )
        quality = max(self.config.corruption.jpeg_quality_min, min(self.config.corruption.jpeg_quality_max, quality))
        from io import BytesIO

        buf = BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _add_noise(self, image: Image.Image, severity: float) -> Image.Image:
        arr = np.asarray(image).astype(np.float32) / 255.0
        sigma = self.config.corruption.noise_std * (0.5 + 0.5 * severity)
        noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8))

    def _color_jitter(self, image: Image.Image, severity: float) -> Image.Image:
        jitter = self.config.corruption.color_jitter * (0.5 + 0.5 * severity)
        if jitter <= 0:
            return image
        def _jitter_factor() -> float:
            return max(0.0, 1.0 + random.uniform(-jitter, jitter))
        image = ImageEnhance.Brightness(image).enhance(_jitter_factor())
        image = ImageEnhance.Contrast(image).enhance(_jitter_factor())
        image = ImageEnhance.Color(image).enhance(_jitter_factor())
        return image

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
        self,
        images: List[Image.Image],
        texts: List[str],
        level: float,
        force: bool = False,
    ) -> Tuple[List[Image.Image], List[str], torch.Tensor, torch.Tensor]:
        cfg = self.config.corruption
        base_level = max(0.0, float(level))
        severity = min(1.0, base_level)
        level = base_level * cfg.max_severity
        corrupted_images: List[Image.Image] = []
        corrupted_texts: List[str] = []
        vis_conf: List[float] = []
        text_conf: List[float] = []

        for image, text in zip(images, texts):
            vis_severity = 0.0
            txt_severity = 0.0
            applied_vis = False
            applied_txt = False
            light_scale = 0.5 * (severity**0.7)
            medium_scale = 0.7 * severity
            heavy_scale = 1.2 * (severity**2.0)

            blur_p = min(1.0, cfg.blur_prob * light_scale)
            motion_p = min(1.0, cfg.motion_blur_prob * light_scale)
            noise_p = min(1.0, cfg.noise_prob * light_scale)
            color_p = min(1.0, cfg.color_prob * light_scale)
            downsample_p = min(1.0, cfg.downsample_prob * medium_scale)
            jpeg_p = min(1.0, cfg.jpeg_prob * medium_scale)
            occlusion_p = min(1.0, cfg.occlusion_prob * heavy_scale)
            crop_p = min(1.0, cfg.crop_prob * heavy_scale)

            if random.random() < blur_p:
                image = self._blur(image, level)
                vis_severity += 0.2
                applied_vis = True
            if random.random() < motion_p:
                image = self._motion_blur(image, level)
                vis_severity += 0.2
                applied_vis = True
            if random.random() < occlusion_p:
                image = self._occlude(image, level)
                vis_severity += 0.4
                applied_vis = True
            if random.random() < crop_p:
                image = self._crop(image, level)
                vis_severity += 0.3
                applied_vis = True
            if random.random() < downsample_p:
                image = self._downsample(image, level)
                vis_severity += 0.15
                applied_vis = True
            if random.random() < jpeg_p:
                image = self._jpeg_compress(image, level)
                vis_severity += 0.15
                applied_vis = True
            if random.random() < noise_p:
                image = self._add_noise(image, level)
                vis_severity += 0.2
                applied_vis = True
            if random.random() < color_p:
                image = self._color_jitter(image, level)
                vis_severity += 0.1
                applied_vis = True

            if random.random() < cfg.text_trunc_prob * level:
                text = self._truncate_text(text, level)
                txt_severity += 0.4
                applied_txt = True
            if random.random() < cfg.text_noise_prob * level:
                text = self._add_text_noise(text, level)
                txt_severity += 0.3
                applied_txt = True

            if force and level > 0.0:
                if not applied_vis and (
                    blur_p
                    + motion_p
                    + occlusion_p
                    + crop_p
                    + downsample_p
                    + jpeg_p
                    + noise_p
                    + color_p
                ) > 0:
                    choices = [
                        "blur",
                        "motion",
                        "occlude",
                        "crop",
                        "downsample",
                        "jpeg",
                        "noise",
                        "color",
                    ]
                    weights = [
                        blur_p,
                        motion_p,
                        occlusion_p,
                        crop_p,
                        downsample_p,
                        jpeg_p,
                        noise_p,
                        color_p,
                    ]
                    op = random.choices(choices, weights=weights, k=1)[0]
                    if op == "blur":
                        image = self._blur(image, level)
                        vis_severity += 0.2
                    elif op == "motion":
                        image = self._motion_blur(image, level)
                        vis_severity += 0.2
                    elif op == "occlude":
                        image = self._occlude(image, level)
                        vis_severity += 0.4
                    else:
                        if op == "crop":
                            image = self._crop(image, level)
                            vis_severity += 0.3
                        elif op == "downsample":
                            image = self._downsample(image, level)
                            vis_severity += 0.15
                        elif op == "jpeg":
                            image = self._jpeg_compress(image, level)
                            vis_severity += 0.15
                        elif op == "noise":
                            image = self._add_noise(image, level)
                            vis_severity += 0.2
                        else:
                            image = self._color_jitter(image, level)
                            vis_severity += 0.1
                if not applied_txt and (cfg.text_trunc_prob + cfg.text_noise_prob) > 0:
                    choices = ["truncate", "noise"]
                    weights = [cfg.text_trunc_prob, cfg.text_noise_prob]
                    op = random.choices(choices, weights=weights, k=1)[0]
                    if op == "truncate":
                        text = self._truncate_text(text, level)
                        txt_severity += 0.4
                    else:
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
            text_weights = torch.nan_to_num(
                text_weights, nan=0.0, posinf=0.0, neginf=0.0
            )
            text_weights = text_weights / text_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled_text = (text_embeds * text_weights.unsqueeze(-1)).sum(dim=1)
        if image_weights is None:
            pooled_image = image_embeds.mean(dim=1)
        else:
            image_weights = torch.nan_to_num(
                image_weights, nan=0.0, posinf=0.0, neginf=0.0
            )
            image_weights = image_weights / image_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled_image = (image_embeds * image_weights.unsqueeze(-1)).sum(dim=1)
        pooled_text = torch.nan_to_num(pooled_text, nan=0.0, posinf=0.0, neginf=0.0)
        pooled_image = torch.nan_to_num(pooled_image, nan=0.0, posinf=0.0, neginf=0.0)
        pooled_text = pooled_text.clamp(-5.0, 5.0)
        pooled_image = pooled_image.clamp(-5.0, 5.0)
        mem_t = torch.tanh(self.text_proj(pooled_text))
        mem_i = torch.tanh(self.image_proj(pooled_image))
        return mem_t, mem_i


class VisualMemoryTokens(nn.Module):
    """Project retrieved image embeddings into memory tokens."""

    def __init__(self, image_dim: int, hidden_dim: int, memory_len: int) -> None:
        super().__init__()
        self.memory_len = memory_len
        self.proj = nn.Linear(image_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        # Small init to stabilize early training for visual memory.
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(
        self, image_embeds: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, top_k, dim = image_embeds.shape
        if weights is None:
            weights = torch.full(
                (batch, top_k), 1.0 / top_k, device=image_embeds.device
            )
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights.clamp_min(0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        take = min(self.memory_len, top_k)
        _, indices = torch.topk(weights, k=take, dim=1)
        gather = indices.unsqueeze(-1).expand(-1, -1, dim)
        selected = torch.gather(image_embeds, 1, gather)
        selected_weights = torch.gather(weights, 1, indices)
        selected = torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)
        selected = selected.clamp(-5.0, 5.0)
        tokens = self.proj(selected) * selected_weights.unsqueeze(-1)
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        tokens = tokens.clamp(-5.0, 5.0)
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

    def sanitize_parameters(self, clamp: Optional[float] = 1.0) -> bool:
        """Replace non-finite parameters in R3 modules to keep training stable."""
        modules = [
            self.memory_aligner,
            self.prefix_enhancer,
            self.visual_memory,
            self.latent_tokens,
            self.gate,
            self.vis_proj,
        ]
        found = False
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                if param is None:
                    continue
                data = param.data
                if not torch.isfinite(data).all():
                    found = True
                    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                if clamp is not None:
                    data = data.clamp(-clamp, clamp)
                param.data.copy_(data)
        return found

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
                    and (
                        self.config.max_text_score < 0
                        or score <= self.config.max_text_score
                    )
                ]
            if image_scores is not None:
                scores = image_scores[idx].tolist()
                imgs = [
                    text
                    for text, score in zip(imgs, scores)
                    if score >= self.config.min_image_score
                    and (
                        self.config.max_image_score < 0
                        or score <= self.config.max_image_score
                    )
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
        self,
        scores: Optional[np.ndarray],
        top_k: int,
        min_score: float,
        max_score: float,
    ) -> torch.Tensor:
        if scores is None:
            return torch.full((top_k,), 1.0 / top_k, device=self.qwen.device)
        weights = torch.tensor(scores, device=self.qwen.device, dtype=torch.float32)
        mask = torch.isfinite(weights)
        if min_score > -1.0:
            mask = mask & (weights >= min_score)
        if max_score > -1.0:
            mask = mask & (weights <= max_score)
        weights = torch.where(
            mask, weights, torch.tensor(float("-inf"), device=weights.device)
        )
        if self.config.use_score_weighting:
            temperature = max(self.config.score_temperature, 1e-6)
            weights = torch.softmax(weights / temperature, dim=-1)
        else:
            weights = mask.float()
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        row_sum = weights.sum(dim=-1, keepdim=True)
        mask_any = mask.any(dim=-1, keepdim=True)
        weights = torch.where(
            mask_any,
            weights / row_sum.clamp_min(1e-6),
            torch.zeros_like(weights),
        )
        return weights

    def _apply_retrieval_confidence(
        self,
        c_vis: torch.Tensor,
        c_text: torch.Tensor,
        text_scores: Optional[np.ndarray],
        image_scores: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        conf_mode = os.getenv("R3_CONF_MODE", "sim").lower()
        if conf_mode not in {"retrieval", "hybrid"}:
            return c_vis, c_text, False
        conf_alpha = float(os.getenv("R3_CONF_ALPHA", "0.5"))

        def _score_conf(scores: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            if scores is None:
                return None
            tensor = torch.as_tensor(scores, device=self.qwen.device, dtype=torch.float32)
            conf = (tensor.mean(dim=1, keepdim=True) + 1.0) / 2.0
            return conf.clamp(0.0, 1.0)

        text_conf = _score_conf(text_scores)
        image_conf = _score_conf(image_scores)
        if conf_mode == "retrieval":
            if text_conf is not None:
                c_text = text_conf
            if image_conf is not None:
                c_vis = image_conf
        else:
            if text_conf is not None:
                c_text = conf_alpha * c_text + (1.0 - conf_alpha) * text_conf
            if image_conf is not None:
                c_vis = conf_alpha * c_vis + (1.0 - conf_alpha) * image_conf
        return c_vis, c_text, True

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
            c_vis, c_text, _ = self._apply_retrieval_confidence(
                c_vis, c_text, text_scores, image_scores
            )

            text_embeds = self._sanitize(text_embeds)
            image_embeds = self._sanitize(image_embeds)
            text_embeds = F.normalize(text_embeds, dim=-1, eps=1e-6).detach()
            image_embeds = F.normalize(image_embeds, dim=-1, eps=1e-6).detach()
            text_weights = (
                self._score_weights(
                    text_scores,
                    top_k,
                    self.config.min_text_score,
                    self.config.max_text_score,
                )
                if text_scores is not None
                else torch.full((len(images), top_k), 1.0 / top_k, device=self.qwen.device)
            )
            image_weights = (
                self._score_weights(
                    image_scores,
                    top_k,
                    self.config.min_image_score,
                    self.config.max_image_score,
                )
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
            mem_t_gate = mem_t.detach()
            mem_i_gate = mem_i.detach()
            vis_feat = self.vis_proj(image_embeds.mean(dim=1))
            vis_feat = self._sanitize(vis_feat)
            if self.gate is not None:
                gates = self.gate(mem_t_gate, mem_i_gate, vis_feat)
                gates = self._sanitize(gates, fill=0.0)
                gates = torch.softmax(gates, dim=-1)
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
        answer_only: bool = False,
    ) -> Any:
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
        text_embeds = self._sanitize(text_embeds)
        image_embeds = self._sanitize(image_embeds)
        text_embeds = F.normalize(text_embeds, dim=-1, eps=1e-6).detach()
        image_embeds = F.normalize(image_embeds, dim=-1, eps=1e-6).detach()
        text_weights = (
            self._score_weights(
                text_scores,
                top_k,
                self.config.min_text_score,
                self.config.max_text_score,
            )
            if text_scores is not None
            else torch.full((len(images), top_k), 1.0 / top_k, device=self.qwen.device)
        )
        image_weights = (
            self._score_weights(
                image_scores,
                top_k,
                self.config.min_image_score,
                self.config.max_image_score,
            )
            if image_scores is not None
            else torch.full((len(images), top_k), 1.0 / top_k, device=self.qwen.device)
        )
        c_vis = c_vis.to(self.qwen.device, dtype=torch.float32)
        c_text = c_text.to(self.qwen.device, dtype=torch.float32)
        c_vis, c_text, use_conf = self._apply_retrieval_confidence(
            c_vis, c_text, text_scores, image_scores
        )
        if self.memory_aligner is not None:
            mem_t, mem_i = self.memory_aligner(
                text_embeds, image_embeds, text_weights=text_weights, image_weights=image_weights
            )
            if use_conf:
                mem_t = mem_t * c_text
                mem_i = mem_i * c_vis
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
        mem_t_gate = mem_t.detach()
        mem_i_gate = mem_i.detach()
        vis_feat = self.vis_proj(image_embeds.mean(dim=1))
        vis_feat = self._sanitize(vis_feat)
        if self.gate is not None:
            gates = self.gate(mem_t_gate, mem_i_gate, vis_feat)
            gates = self._sanitize(gates, fill=0.0)
            gates = torch.softmax(gates, dim=-1)
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
                latent_conf = (
                    (c_vis + c_text) / 2.0
                    if use_conf
                    else torch.ones(len(images), 1, device=self.qwen.device)
                )
                prefix_tokens.append(
                    self.latent_tokens(len(images), latent_conf)
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
                answer_only=answer_only,
            )
            return answers, retrieved_texts, retrieved_image_paths, contexts, prompts
        answers = self.qwen.generate_answer(
            corr_images,
            questions,
            aug_pseudo_texts,
            max_new_tokens=max_new_tokens,
            prefix_embeds=prefix,
            use_soft_prefix=self.config.use_soft_prefix and prefix is not None,
            answer_only=answer_only,
        )
        return answers
