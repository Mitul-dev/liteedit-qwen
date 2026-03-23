"""
data/dataset_local.py
----------------------
Dataset loader for mask-guided local editing.

Recommended public datasets:
  - EditBench (Google) — structured local edit benchmark with masks
  - EMU Edit (Meta) — 10 edit types including local edits
  - MagicBrush — instruction-based local edits with masks
    https://osu-nlp-group.github.io/MagicBrush/

MagicBrush is the best fit for this task — download it from HuggingFace:
    from datasets import load_dataset
    ds = load_dataset("osunlp/MagicBrush")

Key metric for this task: outside_psnr (preservation of unedited regions).
The mask must be precise — a loose mask will artificially inflate outside_psnr.

Quick-start demo: python data/dataset_local.py --build_demo
"""

import os
import argparse
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


LOCAL_EDIT_PROMPTS = [
    "Change the color of the shirt to red",
    "Make the sky look like it is during golden hour",
    "Add a smile to the face",
    "Change the car color to blue",
    "Make the grass greener",
    "Add snow to the roof",
    "Change the wall color to light yellow",
    "Make the water look more turquoise",
    "Change the hair color to blonde",
    "Make the flowers pink instead of white",
]

LOCAL_EDIT_TARGETS = [
    "Only the specified region has changed. The rest of the image is completely "
    "unchanged — same pixels outside the mask boundary. Inside the mask, the "
    "edit is applied cleanly with natural blending at the edges.",
]


def make_soft_mask(size: tuple, center: tuple, radius: int) -> Image.Image:
    """Create a soft circular mask — more realistic than hard rectangles."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = center
    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=255)
    # Soften edges slightly
    mask = mask.filter(ImageFilter.GaussianBlur(radius=8))
    # Re-binarize after blur
    arr = np.array(mask)
    arr = (arr > 128).astype(np.uint8) * 255
    return Image.fromarray(arr)


def make_demo_dataset(output_dir: str, n_images: int = 10):
    """Build a tiny demo dataset for local editing prototyping."""
    for split in ["train", "test"]:
        split_n = n_images if split == "train" else max(3, n_images // 3)
        task_dir  = os.path.join(output_dir, split, "local_edit")
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

            # Create a scene with distinct regions
            img = Image.new("RGB", size, (180, 200, 220))
            draw = ImageDraw.Draw(img)

            # Background elements (should NOT change)
            draw.rectangle([0, 350, 512, 512], fill=(60, 100, 50))   # ground
            for x in range(0, 512, 80):
                draw.rectangle([x+10, 200, x+50, 350], fill=(139, 100, 60))  # trees/posts

            # Subject region (the edit target) — a colored square
            edit_cx, edit_cy = 256, 180
            edit_r = 70
            subject_color = (200, 120, 50)  # orange
            draw.rectangle([
                edit_cx - edit_r, edit_cy - edit_r,
                edit_cx + edit_r, edit_cy + edit_r
            ], fill=subject_color, outline=(150, 80, 20), width=3)

            img.save(os.path.join(input_dir, fname))

            # Reference: same scene but subject region changed to blue
            ref = img.copy()
            draw_r = ImageDraw.Draw(ref)
            draw_r.rectangle([
                edit_cx - edit_r, edit_cy - edit_r,
                edit_cx + edit_r, edit_cy + edit_r
            ], fill=(50, 100, 200), outline=(20, 60, 150), width=3)
            ref.save(os.path.join(ref_dir, fname))

            # Mask: only the subject square region
            mask = make_soft_mask(size, (edit_cx, edit_cy), edit_r + 10)
            mask.save(os.path.join(mask_dir, fname))

            prompt = LOCAL_EDIT_PROMPTS[i % len(LOCAL_EDIT_PROMPTS)]
            target = LOCAL_EDIT_TARGETS[0]
            prompts_lines.append(f"{fname}|{prompt}|{target}")

        with open(os.path.join(task_dir, "prompts.txt"), "w") as f:
            f.write("\n".join(prompts_lines))

        print(f"[dataset_local] Created {split_n} demo images in {task_dir}")


def verify_dataset(data_dir: str, split: str = "train"):
    task_dir  = os.path.join(data_dir, split, "local_edit")
    input_dir = os.path.join(task_dir, "input")
    ref_dir   = os.path.join(task_dir, "ref")
    mask_dir  = os.path.join(task_dir, "mask")

    for d in [input_dir, ref_dir, mask_dir]:
        if not os.path.exists(d):
            print(f"[dataset_local] MISSING: {d}")
            return False

    inputs = set(os.listdir(input_dir))
    masks  = set(os.listdir(mask_dir))
    refs   = set(os.listdir(ref_dir))

    print(f"\n[dataset_local] {split.upper()} split:")
    print(f"  Input images : {len(inputs)}")
    print(f"  References   : {len(refs)}")
    print(f"  Masks        : {len(masks)}")

    # Outside-PSNR sanity check on one sample
    if inputs:
        fname = list(inputs)[0]
        in_img  = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        ref_img = Image.open(os.path.join(ref_dir, fname)).convert("RGB") if fname in refs else None
        msk_img = Image.open(os.path.join(mask_dir, fname)).convert("L") if fname in masks else None

        if ref_img and msk_img:
            in_np   = np.array(in_img).astype(np.float32)
            ref_np  = np.array(ref_img).astype(np.float32)
            msk_np  = np.array(msk_img) / 255.0
            outside = (1 - msk_np)[:, :, None]
            mse = ((in_np - ref_np) ** 2 * outside).sum() / (outside.sum() * 3 + 1e-8)
            outside_psnr = 10 * np.log10(255**2 / (mse + 1e-8))
            print(f"  Outside-PSNR check (sample): {outside_psnr:.2f} dB")
            print(f"  (Should be high if mask is tight and ref only edits inside mask)")

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
