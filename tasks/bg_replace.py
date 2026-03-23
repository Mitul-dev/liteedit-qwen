"""
tasks/bg_replace.py
-------------------
Background replacement task.

Strategy:
  - No mask needed (model infers foreground/background from prompt)
  - Prompt engineering guides the model to replace only the background
  - We ask Qwen to generate a new image description then use it for editing

Ablation notes:
  - Baseline: fp16, 50 steps, no LoRA, no mask gating
  - Quantized: INT4 — measure LPIPS vs baseline
  - Low-step: 10 steps — bg replacement tolerates this best of the 3 tasks
  - LoRA: fine-tuned on bg-replace pairs — should recover quality lost to INT4
"""

from PIL import Image
import numpy as np
import torch
from tasks.base_task import BaseTask
from models.mask_attention import MaskAttentionController


SYSTEM_PROMPT = (
    "You are an image editing assistant. When asked to replace a background, "
    "keep all foreground subjects (people, objects) exactly unchanged. "
    "Only modify what is clearly the background. "
    "Output: describe the edited image in detail, then provide editing instructions."
)


class BgReplaceTask(BaseTask):

    def __init__(self, model, processor, config: dict):
        super().__init__(model, processor, config)
        self.use_mask_attention = config.get("mask_attention", False)
        if self.use_mask_attention:
            self.controller = MaskAttentionController(model)

    def build_messages(self, image: Image.Image, prompt: str) -> list:
        task_prompt = (
            f"Please edit this image: {prompt}. "
            "Keep all foreground subjects exactly the same. "
            "Only replace the background region. "
            "Describe what the edited image should look like."
        )
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": task_prompt},
                ],
            },
        ]

    def run(
        self,
        image: Image.Image,
        prompt: str,
        mask: Image.Image = None,
    ) -> Image.Image:
        """
        Run background replacement.

        Args:
            image:  Input image
            prompt: e.g. "replace the background with a snowy mountain landscape"
            mask:   Optional — if provided, used for mask-guided attention gating

        Returns:
            Edited image (PIL)
        """
        image = self.preprocess_image(image)

        # Optionally enable mask-guided attention
        if self.use_mask_attention and mask is not None:
            self.controller.set_mask(mask, image_size=(self.image_size, self.image_size))
            self.controller.enable()

        # Generate edit description
        edit_description = self.generate(image, prompt)

        if self.use_mask_attention and mask is not None:
            self.controller.disable()
            self.controller.clear_hooks()

        # For the prototype: return the original image with edit description
        # overlaid as metadata. In a full pipeline, this description would feed
        # a diffusion inpainting model. This scaffold lets you measure
        # text-quality metrics before connecting the diffusion backend.
        edited_image = self._apply_edit(image, edit_description, mask)
        return edited_image, edit_description

    def _apply_edit(
        self,
        image: Image.Image,
        description: str,
        mask: Image.Image = None,
    ) -> Image.Image:
        """
        Placeholder for the diffusion backend.
        Currently returns the original image so the eval pipeline works
        end-to-end while you integrate a diffusion model.

        Replace this method body with your actual inpainting call, e.g.:
            from diffusers import StableDiffusionInpaintPipeline
            pipe = StableDiffusionInpaintPipeline.from_pretrained(...)
            result = pipe(prompt=description, image=image, mask_image=mask)
            return result.images[0]
        """
        # TODO: replace with actual diffusion inpainting call
        print(f"[bg_replace] Edit description: {description[:120]}...")
        return image
