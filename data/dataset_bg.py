"""
data/dataset_bg.py
------------------
Dataset loader for background replacement.

Recommended public datasets:
  - COCO (https://cocodataset.org) — use person/object segments as foreground
  - BG-20k (https://github.com/JizhiziLi/GFM) — high-quality background matting
  - Open Images v7 — diverse scenes, good for background variety

Quick-start dataset for prototyping (no download needed):
  Run: python data/dataset_bg.py --build_demo
  This creates a tiny 10-image demo set using HuggingFace datasets.

Expected folder layout after download/preparation:
    data/
      train/
        bg_replace/
          input/        ← original images (subject on original background)
          ref/          ← ground truth (same subject, different background)
          prompts.txt   ← filename|user_prompt|target_description
      test/
        bg_replace/     ← same structure, separate split
"""

import os
import argparse
from PIL import Image, ImageDraw
import numpy as np


# ------------------------------------------------------------------
# Prompt templates for background replacement
# Vary these to improve LoRA generalisation
# ------------------------------------------------------------------

BG_PROMPTS = [
    "Replace the background with a sunny beach with blue ocean waves",
    "Change the background to a snowy mountain landscape at sunset",
    "Replace the background with a modern city skyline at night",
    "Change the background to a lush green forest with morning fog",
    "Replace the background with a cozy indoor library with warm lighting",
    "Change the background to a space scene with stars and nebulae",
    "Replace the background with a rustic countryside with golden fields",
    "Change the background to a rainy city street with reflections",
    "Replace the background with a tropical rainforest",
    "Change the background to a minimalist white studio backdrop",
]

BG_TARGET_TEMPLATES = [
    "The image shows the same subject in the foreground, now placed in front of {bg}. "
    "The background has been completely replaced while the subject remains identical. "
    "Lighting and shadows are consistent with the new environment.",
]


def make_demo_dataset(output_dir: str, n_images: int = 10):
    """
    Build a tiny demo dataset for prototyping without downloading large datasets.
    Creates synthetic images with colored foreground subjects on plain backgrounds.
    Uses only PIL — no internet required.
    """
    for split in ["train", "test"]:
        split_n = n_images if split == "train" else max(3, n_images // 3)
        task_dir = os.path.join(output_dir, split, "bg_replace")
        input_dir = os.path.join(task_dir, "input")
        ref_dir   = os.path.join(task_dir, "ref")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(ref_dir,   exist_ok=True)

        prompts_lines = []

        for i in range(split_n):
            fname = f"img_{i:04d}.png"

            # Input: colored circle (subject) on gradient background
            img = Image.new("RGB", (512, 512), color=(200, 220, 240))
            draw = ImageDraw.Draw(img)
            # Gradient-ish background
            for y in range(512):
                c = int(200 - y * 0.3)
                draw.line([(0, y), (512, y)], fill=(c, c+20, 255-c//2))
            # Subject: red/orange circle
            cx, cy, r = 256, 220, 100
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(220, 80, 60), outline=(180, 40, 20), width=3)
            img.save(os.path.join(input_dir, fname))

            # Ref: same circle on different (green) background
            ref = Image.new("RGB", (512, 512))
            draw_r = ImageDraw.Draw(ref)
            for y in range(512):
                c = int(100 + y * 0.15)
                draw_r.line([(0, y), (512, y)], fill=(30, c, 40))
            draw_r.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(220, 80, 60), outline=(180, 40, 20), width=3)
            ref.save(os.path.join(ref_dir, fname))

            prompt = BG_PROMPTS[i % len(BG_PROMPTS)]
            target = (
                f"The subject (a red-orange circular shape) remains unchanged in the center. "
                f"The background has been replaced with: {prompt.lower().replace('replace the background with ', '').replace('change the background to ', '')}. "
                f"Colors and lighting are adapted to the new scene."
            )
            prompts_lines.append(f"{fname}|{prompt}|{target}")

        with open(os.path.join(task_dir, "prompts.txt"), "w") as f:
            f.write("\n".join(prompts_lines))

        print(f"[dataset_bg] Created {split_n} demo images in {task_dir}")


def verify_dataset(data_dir: str, split: str = "train"):
    """Check dataset integrity and print a summary."""
    task_dir  = os.path.join(data_dir, split, "bg_replace")
    input_dir = os.path.join(task_dir, "input")
    ref_dir   = os.path.join(task_dir, "ref")
    prompt_f  = os.path.join(task_dir, "prompts.txt")

    issues = []
    if not os.path.exists(input_dir):
        issues.append(f"Missing input dir: {input_dir}")
    if not os.path.exists(ref_dir):
        issues.append(f"Missing ref dir: {ref_dir}")
    if not os.path.exists(prompt_f):
        issues.append(f"Missing prompts.txt: {prompt_f}")

    if issues:
        for issue in issues:
            print(f"[dataset_bg] ERROR: {issue}")
        return False

    input_files  = set(os.listdir(input_dir))
    ref_files    = set(os.listdir(ref_dir))
    missing_refs = input_files - ref_files

    with open(prompt_f) as f:
        prompt_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    prompt_fnames = {l.split("|")[0].strip() for l in prompt_lines if "|" in l}
    missing_prompts = input_files - prompt_fnames

    print(f"\n[dataset_bg] {split.upper()} split summary:")
    print(f"  Input images  : {len(input_files)}")
    print(f"  Reference imgs: {len(ref_files)}")
    print(f"  Prompt entries: {len(prompt_lines)}")
    if missing_refs:
        print(f"  WARNING: {len(missing_refs)} inputs missing references: {list(missing_refs)[:5]}")
    if missing_prompts:
        print(f"  WARNING: {len(missing_prompts)} inputs missing prompts: {list(missing_prompts)[:5]}")

    return len(issues) == 0 and len(missing_refs) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_demo", action="store_true",
                        help="Build a tiny demo dataset for prototyping")
    parser.add_argument("--data_dir", default="data",
                        help="Root data directory")
    parser.add_argument("--n_images", type=int, default=10)
    parser.add_argument("--verify", action="store_true",
                        help="Verify an existing dataset")
    args = parser.parse_args()

    if args.build_demo:
        make_demo_dataset(args.data_dir, args.n_images)
        verify_dataset(args.data_dir, "train")
        verify_dataset(args.data_dir, "test")
    elif args.verify:
        verify_dataset(args.data_dir, "train")
        verify_dataset(args.data_dir, "test")
    else:
        parser.print_help()
