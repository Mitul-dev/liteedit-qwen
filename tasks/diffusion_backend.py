"""
tasks/diffusion_backend.py
--------------------------
Shared diffusion inpainting backend used by all three tasks.

Uses Stable Diffusion Inpainting (runwayml/stable-diffusion-inpainting).
Loaded once and reused across tasks to avoid reloading weights each call.

Key design:
  - Qwen generates the edit description (what to paint)
  - SD inpainting does the actual pixel editing (where and how to paint)
  - For bg_replace: auto-generate a background mask from the image
  - For person_remove: use the provided binary mask directly
  - For local_edit: use the provided mask + hard-composite outside pixels

Usage:
    from tasks.diffusion_backend import get_diffusion_pipe, run_inpaint

    pipe = get_diffusion_pipe(num_steps=50, device="cuda")
    result = run_inpaint(pipe, image, mask, prompt, num_steps=50)
"""

import torch
import numpy as np
from PIL import Image, ImageFilter

_PIPE = None  # module-level singleton — loaded once, reused


def get_diffusion_pipe(num_steps: int = 50, device: str = "cuda"):
    """
    Load and return the SD inpainting pipeline.
    Cached as a module-level singleton so it's only loaded once per session.
    """
    global _PIPE

    if _PIPE is not None:
        return _PIPE

    from diffusers import StableDiffusionInpaintPipeline

    print("[diffusion] Loading SD inpainting pipeline...")
    _PIPE = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,        # disable for research use
        requires_safety_checker=False,
    )
    _PIPE = _PIPE.to(device)
    _PIPE.set_progress_bar_config(disable=True)  # cleaner output during eval
    print("[diffusion] Pipeline ready.")
    return _PIPE


def make_background_mask(image: Image.Image, blur_radius: int = 8) -> Image.Image:
    """
    Auto-generate a rough background mask for bg_replace.
    Uses edge detection to find the foreground boundary,
    then fills the outside as the background region.

    This is a simple heuristic — a SAM-based mask would be better,
    but this works without any additional models.
    """
    img_np = np.array(image.convert("L"))

    # Sobel-like edge detection using PIL
    from PIL import ImageFilter
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edges_np = np.array(edges)

    # Threshold edges to get foreground boundary
    threshold = edges_np.max() * 0.15
    fg_mask = (edges_np > threshold).astype(np.uint8) * 255

    # Dilate slightly to cover the full foreground
    fg_pil = Image.fromarray(fg_mask)
    for _ in range(3):
        fg_pil = fg_pil.filter(ImageFilter.MaxFilter(5))

    # Invert: background = white (region to replace)
    bg_mask = Image.fromarray(255 - np.array(fg_pil))
    bg_mask = bg_mask.filter(ImageFilter.GaussianBlur(blur_radius))

    # Re-binarize after blur
    arr = np.array(bg_mask)
    arr = (arr > 128).astype(np.uint8) * 255
    return Image.fromarray(arr)


def run_inpaint(
    pipe,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    target_size: int = 512,
) -> Image.Image:
    """
    Run one SD inpainting forward pass.

    Args:
        pipe:          StableDiffusionInpaintPipeline
        image:         Input PIL image (RGB)
        mask:          Binary mask — white = region to edit
        prompt:        Text description of what to generate in the masked region
        num_steps:     Diffusion steps (50 = quality, 10 = fast)
        guidance_scale: CFG scale (7.5 is standard)
        target_size:   Resize to this before inpainting (SD works best at 512)

    Returns:
        Edited PIL image resized back to original input size
    """
    original_size = image.size

    # SD inpainting requires 512x512
    image_resized = image.resize((target_size, target_size), Image.LANCZOS)
    mask_resized  = mask.resize((target_size, target_size), Image.NEAREST)

    # Ensure mask is binary L mode
    mask_resized = mask_resized.convert("L")
    arr = np.array(mask_resized)
    arr = (arr > 128).astype(np.uint8) * 255
    mask_resized = Image.fromarray(arr)

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )

    edited = result.images[0]

    # Resize back to original dimensions
    edited = edited.resize(original_size, Image.LANCZOS)
    return edited


def hard_composite(
    original: Image.Image,
    edited: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """
    Hard-composite edited image with original outside the mask.
    Critical for local_edit — ensures pixels outside mask are
    IDENTICAL to the original (maximises outside_psnr metric).

    Inside mask  → use edited pixels
    Outside mask → use original pixels exactly
    """
    orig_np   = np.array(original.convert("RGB")).astype(np.float32)
    edited_np = np.array(edited.convert("RGB").resize(original.size)).astype(np.float32)
    mask_np   = np.array(mask.convert("L").resize(original.size)) / 255.0
    mask_3ch  = np.stack([mask_np] * 3, axis=-1)

    result = edited_np * mask_3ch + orig_np * (1 - mask_3ch)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))
