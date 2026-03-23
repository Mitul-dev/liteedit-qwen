"""
tasks/bg_replace.py
-------------------
Background replacement task.

Pipeline:
  1. Qwen2.5-VL generates a detailed description of the new background
  2. Auto-generate background mask (edge detection heuristic)
  3. SD inpainting fills the background region using the description
"""

from PIL import Image
import torch
from tasks.base_task import BaseTask
from tasks.diffusion_backend import get_diffusion_pipe, make_background_mask, run_inpaint
from models.mask_attention import MaskAttentionController


SYSTEM_PROMPT = (
    "You are an image editing assistant. When asked to replace a background, "
    "keep all foreground subjects (people, objects) exactly unchanged. "
    "Only describe the new background in vivid detail — colors, textures, "
    "lighting, atmosphere. Be specific and descriptive. "
    "Output only the background description, nothing else."
)


class BgReplaceTask(BaseTask):

    def __init__(self, model, processor, config: dict):
        super().__init__(model, processor, config)
        self.use_mask_attention = config.get("mask_attention", False)
        self.num_steps = config.get("num_steps", 50)
        self._pipe = None  # lazy-loaded on first run
        if self.use_mask_attention:
            self.controller = MaskAttentionController(model)

    def build_messages(self, image: Image.Image, prompt: str) -> list:
        task_prompt = (
            f"Task: {prompt}. "
            "Describe only the new background in vivid detail — "
            "what it looks like, colors, textures, lighting, atmosphere. "
            "Keep foreground subjects unchanged. "
            "Be specific: this description will be used to generate the new background."
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

        # Optional mask-guided attention gating
        if self.use_mask_attention and mask is not None:
            self.controller.set_mask(mask, image_size=(self.image_size, self.image_size))
            self.controller.enable()

        # Step 1: Qwen generates background description
        bg_description = self.generate(image, prompt)

        if self.use_mask_attention and mask is not None:
            self.controller.disable()
            self.controller.clear_hooks()

        # Step 2: apply diffusion inpainting
        edited = self._apply_edit(image, bg_description, mask)
        return edited, bg_description

    def _apply_edit(
        self,
        image: Image.Image,
        description: str,
        mask: Image.Image = None,
    ) -> Image.Image:
        print(f"[bg_replace] Description: {description[:100]}...")

        # Auto-generate background mask if not provided
        if mask is None:
            mask = make_background_mask(image)

        pipe = self._get_pipe()
        edited = run_inpaint(
            pipe=pipe,
            image=image,
            mask=mask,
            prompt=description,
            num_steps=self.num_steps,
        )
        return edited
