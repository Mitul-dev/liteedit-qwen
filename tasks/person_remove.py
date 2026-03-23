"""
tasks/person_remove.py
-----------------------
Person removal task.

Pipeline:
  1. Qwen2.5-VL describes what should fill the removed region (scene context)
  2. SD inpainting fills the masked region using the description
  3. No hard composite needed — the whole mask region should change
"""

from PIL import Image
import numpy as np
from tasks.base_task import BaseTask
from tasks.diffusion_backend import get_diffusion_pipe, run_inpaint
from models.mask_attention import MaskAttentionController


SYSTEM_PROMPT = (
    "You are an image editing assistant specializing in object removal. "
    "Describe what the background should look like after the person is removed. "
    "Focus on: what surface or scenery would be visible behind where they stood, "
    "matching colors, textures, and lighting of the surrounding area. "
    "Output only the fill description — no preamble, no explanation."
)


class PersonRemoveTask(BaseTask):

    def __init__(self, model, processor, config: dict):
        super().__init__(model, processor, config)
        self.use_mask_attention = config.get("mask_attention", False)
        self.num_steps = config.get("num_steps", 50)
        self._pipe = None
        if self.use_mask_attention:
            self.controller = MaskAttentionController(model)

    def build_messages(self, image: Image.Image, prompt: str) -> list:
        task_prompt = (
            f"{prompt}. "
            "Describe in detail what the background should look like after removal — "
            "texture, color, lighting, and any objects that would naturally be visible. "
            "Match the style and lighting of the rest of the scene exactly."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": task_prompt},
                ],
            },
        ]

    def _get_pipe(self):
        if self._pipe is None:
            self._pipe = get_diffusion_pipe(
                num_steps=self.num_steps,
                device="cuda",
            )
        return self._pipe

    def run(
        self,
        image: Image.Image,
        prompt: str,
        mask: Image.Image = None,
    ):
        image = self.preprocess_image(image)

        if mask is None:
            print("[person_remove] WARNING: No mask provided. Results will be poor.")

        # Mask-guided attention gating
        if self.use_mask_attention and mask is not None:
            self.controller.set_mask(mask, image_size=(self.image_size, self.image_size))
            self.controller.enable()

        # Step 1: Qwen describes the fill content
        fill_description = self.generate(image, prompt)

        if self.use_mask_attention and mask is not None:
            self.controller.disable()
            self.controller.clear_hooks()

        # Step 2: SD inpainting fills the person region
        edited = self._apply_inpaint(image, fill_description, mask)
        return edited, fill_description

    def _apply_inpaint(
        self,
        image: Image.Image,
        description: str,
        mask: Image.Image = None,
    ) -> Image.Image:
        print(f"[person_remove] Fill: {description[:100]}...")

        if mask is None:
            return image

        pipe = self._get_pipe()
        edited = run_inpaint(
            pipe=pipe,
            image=image,
            mask=mask,
            prompt=description,
            num_steps=self.num_steps,
        )
        return edited
