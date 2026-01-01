#!/usr/bin/env python3
"""Preview corruption effects for image/text inputs."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

from PIL import Image
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.train_config import R3Config
from models.r3_modules import CorruptionSimulator


def _parse_levels(raw: str) -> List[float]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("levels cannot be empty")
    return values


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview R3 corruption effects.")
    parser.add_argument("--image", default='data/image.jpg', help="Path to an image file.")
    parser.add_argument("--text", default='“Exercise 10 of 12” screen with a lunge illustration, exercise label (“LUNGES”), and 30-second timer.', help="Input text to corrupt.")
    parser.add_argument(
        "--levels",
        default="0,0.2,0.4,0.6,0.8",
        help="Comma-separated corruption levels in [0,1].",
    )
    parser.add_argument("--out_dir", default="outputs/corruption_preview")
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force at least one corruption per modality for preview.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for corruption. Omit or set <0 to randomize each run.",
    )
    parser.add_argument("--max_severity", type=float, default=None)
    parser.add_argument("--blur_prob", type=float, default=None)
    parser.add_argument("--motion_blur_prob", type=float, default=None)
    parser.add_argument("--occlusion_prob", type=float, default=None)
    parser.add_argument("--crop_prob", type=float, default=None)
    parser.add_argument("--downsample_prob", type=float, default=None)
    parser.add_argument("--jpeg_prob", type=float, default=None)
    parser.add_argument("--noise_prob", type=float, default=None)
    parser.add_argument("--color_prob", type=float, default=None)
    parser.add_argument("--text_trunc_prob", type=float, default=None)
    parser.add_argument("--text_noise_prob", type=float, default=None)
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument("--jpeg_quality_min", type=int, default=None)
    parser.add_argument("--jpeg_quality_max", type=int, default=None)
    parser.add_argument("--color_jitter", type=float, default=None)
    parser.add_argument("--disable_image", action="store_true")
    parser.add_argument("--disable_text", action="store_true")
    args = parser.parse_args()

    if args.image is None and args.text is None:
        raise ValueError("Provide at least --image or --text.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = R3Config()
    if args.max_severity is not None:
        cfg.corruption.max_severity = args.max_severity
    if args.blur_prob is not None:
        cfg.corruption.blur_prob = args.blur_prob
    if args.motion_blur_prob is not None:
        cfg.corruption.motion_blur_prob = args.motion_blur_prob
    if args.occlusion_prob is not None:
        cfg.corruption.occlusion_prob = args.occlusion_prob
    if args.crop_prob is not None:
        cfg.corruption.crop_prob = args.crop_prob
    if args.downsample_prob is not None:
        cfg.corruption.downsample_prob = args.downsample_prob
    if args.jpeg_prob is not None:
        cfg.corruption.jpeg_prob = args.jpeg_prob
    if args.noise_prob is not None:
        cfg.corruption.noise_prob = args.noise_prob
    if args.color_prob is not None:
        cfg.corruption.color_prob = args.color_prob
    if args.text_trunc_prob is not None:
        cfg.corruption.text_trunc_prob = args.text_trunc_prob
    if args.text_noise_prob is not None:
        cfg.corruption.text_noise_prob = args.text_noise_prob
    if args.noise_std is not None:
        cfg.corruption.noise_std = args.noise_std
    if args.jpeg_quality_min is not None:
        cfg.corruption.jpeg_quality_min = args.jpeg_quality_min
    if args.jpeg_quality_max is not None:
        cfg.corruption.jpeg_quality_max = args.jpeg_quality_max
    if args.color_jitter is not None:
        cfg.corruption.color_jitter = args.color_jitter

    if args.disable_image:
        cfg.corruption.blur_prob = 0.0
        cfg.corruption.motion_blur_prob = 0.0
        cfg.corruption.occlusion_prob = 0.0
        cfg.corruption.crop_prob = 0.0
        cfg.corruption.downsample_prob = 0.0
        cfg.corruption.jpeg_prob = 0.0
        cfg.corruption.noise_prob = 0.0
        cfg.corruption.color_prob = 0.0
    if args.disable_text:
        cfg.corruption.text_trunc_prob = 0.0
        cfg.corruption.text_noise_prob = 0.0

    image = _load_image(args.image) if args.image else Image.new("RGB", (512, 512), color=(0, 0, 0))
    text = args.text or ""

    image.save(os.path.join(args.out_dir, "original.png"))
    with open(os.path.join(args.out_dir, "original.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    simulator = CorruptionSimulator(cfg)
    levels = _parse_levels(args.levels)

    import random
    seed = args.seed
    if seed is None or seed < 0:
        seed = int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed)
    logging.info("Using seed %d", seed)

    for level in levels:
        level = max(0.0, float(level))
        effective = level * cfg.corruption.max_severity
        corrupted_images, corrupted_texts, c_vis, c_text = simulator(
            [image], [text], level, force=args.force
        )
        tag = f"{level:.2f}".replace(".", "_")
        out_image = os.path.join(args.out_dir, f"corrupt_{tag}.png")
        out_text = os.path.join(args.out_dir, f"corrupt_{tag}.txt")
        corrupted_images[0].save(out_image)
        with open(out_text, "w", encoding="utf-8") as f:
            f.write(corrupted_texts[0])
        logging.info(
            "Level %.2f (effective %.2f): image=%s text=%s c_vis=%.3f c_text=%.3f",
            level,
            effective,
            out_image,
            out_text,
            float(c_vis.squeeze().item()),
            float(c_text.squeeze().item()),
        )


if __name__ == "__main__":
    main()
