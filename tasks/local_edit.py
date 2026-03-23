"""
tasks/local_edit.py
--------------------
Mask-guided local editing task.

Pipeline:
  1. Qwen2.5-VL describes what the edit should look like inside the mask
  2. SD inpainting edits only the masked region
  3. Hard composite: pixels outside mask are REPLACED with original exactly
     (this maximises outside_psnr — the key metric for this task)
"""

from PIL import Image
import numpy as np
from tasks.base_task import BaseTask
from tasks.diffusion_backend import get_diffusion_pipe, run_inpaint, hard_composite
from models.mask_attention import MaskAttentionController


SYSTEM_PROMPT = (
    "You are a precise image editing assistant. "
    "Edit ONLY the region inside the mask. Everything outside must stay identical. "
    "Describe exactly what the masked region should look like after the edit — "
    "colors, textures, style. Be specific and concise. "
    "Output only the edit description for the masked region."
)


class LocalEditTask(BaseTask):

    def __init__(self, model, processor, config: dict):
        super().__init__(model, processor, config)
        self.use_mask_attention = config.get("mask_attention", False)
        self.num_steps = config.get("num_steps", 50)
        self._pipe = None
        if self.use_mask_attention:
            self.controller = MaskAttentionController(model)

    def build_messages(self, image: Image.Image, prompt: str) -> list:
        task_prompt = (
            f"Edit only the highlighted/masked region: {prompt}. "
            "Describe what the masked region should look like after the edit. "
            "Be specific about colors, textures, and style. "
            "Do NOT describe anything outside the mask."
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
            raise ValueError("[local_edit] Mask is required for local editing.")

        # Mask-guided attention gating — most important for this task
        if self.use_mask_attention:
            self.controller.set_mask(mask, image_size=(self.image_size, self.image_size))
            self.controller.enable()

        # Step 1: Qwen describes the local edit
        edit_description = self.generate(image, prompt)

        if self.use_mask_attention:
            self.controller.disable()
            self.controller.clear_hooks()

        # Step 2: SD inpainting + hard composite
        edited = self._apply_local_edit(image, edit_description, mask)
        return edited, edit_description

    def _apply_local_edit(
        self,
        image: Image.Image,
        description: str,
        mask: Image.Image,
    ) -> Image.Image:
        print(f"[local_edit] Edit: {description[:100]}...")

        pipe = self._get_pipe()

        # Run inpainting
        edited = run_inpaint(
            pipe=pipe,
            image=image,
            mask=mask,
            prompt=description,
            num_steps=self.num_steps,
        )

        # Hard composite: restore original pixels outside the mask
        # This is critical — outside_psnr measures exactly this preservation
        edited = hard_composite(
            original=image,
            edited=edited,
            mask=mask,
        )

        return edited

    def compute_outside_preservation(
        self,
        original: Image.Image,
        edited: Image.Image,
        mask: Image.Image,
    ) -> float:
        """PSNR of unmasked region. Higher = better preservation."""
        orig_np  = np.array(original.convert("RGB")).astype(np.float32)
        edit_np  = np.array(edited.convert("RGB")).astype(np.float32)
        mask_np  = np.array(mask.convert("L").resize(original.size)) / 255.0
        outside  = (1 - mask_np)[:, :, None]
        mse = ((orig_np - edit_np) ** 2 * outside).sum() / (outside.sum() * 3 + 1e-8)
        return float(10 * np.log10(255**2 / (mse + 1e-8)))
