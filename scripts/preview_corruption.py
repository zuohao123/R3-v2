#!/usr/bin/env python3
"""Preview corruption effects for image/text inputs."""
from __future__ import annotations

import argparse
import logging
import os
from typing import List

from PIL import Image

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
    parser.add_argument("--text", default='What is the application name?', help="Input text to corrupt.")
    parser.add_argument(
        "--levels",
        default="0,0.2,0.4,0.6,0.8",
        help="Comma-separated corruption levels in [0,1].",
    )
    parser.add_argument("--out_dir", default="outputs/corruption_preview")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_severity", type=float, default=None)
    parser.add_argument("--blur_prob", type=float, default=None)
    parser.add_argument("--occlusion_prob", type=float, default=None)
    parser.add_argument("--crop_prob", type=float, default=None)
    parser.add_argument("--text_trunc_prob", type=float, default=None)
    parser.add_argument("--text_noise_prob", type=float, default=None)
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
    if args.occlusion_prob is not None:
        cfg.corruption.occlusion_prob = args.occlusion_prob
    if args.crop_prob is not None:
        cfg.corruption.crop_prob = args.crop_prob
    if args.text_trunc_prob is not None:
        cfg.corruption.text_trunc_prob = args.text_trunc_prob
    if args.text_noise_prob is not None:
        cfg.corruption.text_noise_prob = args.text_noise_prob

    if args.disable_image:
        cfg.corruption.blur_prob = 0.0
        cfg.corruption.occlusion_prob = 0.0
        cfg.corruption.crop_prob = 0.0
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
    random.seed(args.seed)

    for level in levels:
        level = max(0.0, min(1.0, float(level)))
        effective = level * cfg.corruption.max_severity
        corrupted_images, corrupted_texts, c_vis, c_text = simulator(
            [image], [text], effective
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
