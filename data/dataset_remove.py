"""
data/dataset_remove.py
-----------------------
Dataset loader for person removal.

Recommended public datasets:
  - INRIA Person Dataset — people with diverse backgrounds
  - CrowdHuman — crowded scenes, good for multi-person removal
  - MOT17/MOT20 — tracking datasets with clean person annotations
  - Or: use COCO with person segmentation masks directly

Mask format: white (255) = person region to remove, black (0) = preserve

Quick-start demo: python data/dataset_remove.py --build_demo
"""

import os
import argparse
from PIL import Image, ImageDraw
import numpy as np


REMOVE_PROMPTS = [
    "Remove the person standing in the center of the image",
    "Erase the person on the left side and fill with background",
    "Remove all people from the scene",
    "Delete the person in the foreground",
    "Remove the person and restore the background behind them",
    "Erase the standing figure and fill the area naturally",
    "Remove the person on the right and complete the background",
    "Delete the person walking in the scene",
]

REMOVE_TARGETS = [
    "The person has been removed from the image. The area where they stood now shows "
    "the background continuing naturally — same texture, lighting, and perspective as "
    "the surrounding area. No ghosting or artifacts remain.",
    "The person is gone. The background fills the space coherently with matching "
    "colors, textures, and lighting. The scene looks as if the person was never there.",
]


def make_synthetic_mask(size: tuple, person_bbox: tuple) -> Image.Image:
    """Create a binary mask with a person-shaped region (rectangle for demo)."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(person_bbox, fill=255)
    return mask


def make_demo_dataset(output_dir: str, n_images: int = 10):
    """Build a tiny demo dataset for person removal prototyping."""
    for split in ["train", "test"]:
        split_n = n_images if split == "train" else max(3, n_images // 3)
        task_dir  = os.path.join(output_dir, split, "person_remove")
        input_dir = os.path.join(task_dir, "input")
        ref_dir   = os.path.join(task_dir, "ref")
        mask_dir  = os.path.join(task_dir, "mask")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(ref_dir,   exist_ok=True)
        os.makedirs(mask_dir,  exist_ok=True)

        prompts_lines = []

        for i in range(split_n):
            fname = f"img_{i:04d}.png"
            size  = (512, 512)

            # Background: gradient scene
            bg = Image.new("RGB", size)
            draw = ImageDraw.Draw(bg)
            for y in range(512):
                sky_c = int(135 + y * 0.1)
                draw.line([(0, y), (512, y)], fill=(100, sky_c, 200))
            # Ground
            draw.rectangle([0, 350, 512, 512], fill=(80, 120, 60))

            # Person: simple rectangle silhouette
            px1, py1, px2, py2 = 180, 150, 260, 380
            person_bbox = (px1, py1, px2, py2)

            # Input: background + person
            input_img = bg.copy()
            draw_i = ImageDraw.Draw(input_img)
            draw_i.rectangle(person_bbox, fill=(50, 50, 150))    # body
            draw_i.ellipse([200, 110, 240, 155], fill=(210, 170, 130))  # head
            input_img.save(os.path.join(input_dir, fname))

            # Ref: background only (person removed)
            bg.save(os.path.join(ref_dir, fname))

            # Mask: white = person region
            mask = make_synthetic_mask(size, (px1-10, py1-50, px2+10, py2+5))
            mask.save(os.path.join(mask_dir, fname))

            prompt = REMOVE_PROMPTS[i % len(REMOVE_PROMPTS)]
            target = REMOVE_TARGETS[i % len(REMOVE_TARGETS)]
            prompts_lines.append(f"{fname}|{prompt}|{target}")

        with open(os.path.join(task_dir, "prompts.txt"), "w") as f:
            f.write("\n".join(prompts_lines))

        print(f"[dataset_remove] Created {split_n} demo images in {task_dir}")


def verify_dataset(data_dir: str, split: str = "train"):
    task_dir  = os.path.join(data_dir, split, "person_remove")
    input_dir = os.path.join(task_dir, "input")
    ref_dir   = os.path.join(task_dir, "ref")
    mask_dir  = os.path.join(task_dir, "mask")
    prompt_f  = os.path.join(task_dir, "prompts.txt")

    for d in [input_dir, ref_dir, mask_dir, prompt_f]:
        if not os.path.exists(d):
            print(f"[dataset_remove] MISSING: {d}")
            return False

    inputs = set(os.listdir(input_dir))
    masks  = set(os.listdir(mask_dir))
    print(f"\n[dataset_remove] {split.upper()} split:")
    print(f"  Input images : {len(inputs)}")
    print(f"  Masks        : {len(masks)}")
    missing_masks = inputs - masks
    if missing_masks:
        print(f"  WARNING: {len(missing_masks)} inputs missing masks")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_demo", action="store_true")
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--n_images",   type=int, default=10)
    parser.add_argument("--verify",     action="store_true")
    args = parser.parse_args()

    if args.build_demo:
        make_demo_dataset(args.data_dir, args.n_images)
    elif args.verify:
        verify_dataset(args.data_dir, "train")
        verify_dataset(args.data_dir, "test")
    else:
        parser.print_help()
