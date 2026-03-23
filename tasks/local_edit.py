"""
tasks/local_edit.py
--------------------
Mask-guided local editing task.

Strategy:
  - Mask defines exact region to edit; everything outside must be pixel-stable
  - Mask-guided attention gating is CRITICAL here — this is where it matters most
  - Prompt specifies what to change within the masked region only

Ablation notes:
  - This task shows the biggest quality difference with vs without mask gating
  - Low-step degrades most sharply here (precision task)
  - LoRA fine-tuned on local edits should show clearest quality recovery
  - Key paper metric: PSNR of unmasked region (measures how well outside is preserved)
"""

from PIL import Image
import numpy as np
from tasks.base_task import BaseTask
from models.mask_attention import MaskAttentionController


SYSTEM_PROMPT = (
    "You are a precise image editing assistant. You edit ONLY the region "
    "specified by the user's mask or description. Everything outside the "
    "specified region must remain completely unchanged — same pixels, same colors, "
    "same texture. Your edits should blend seamlessly with the surrounding image."
)


class LocalEditTask(BaseTask):

    def __init__(self, model, processor, config: dict):
        super().__init__(model, processor, config)
        self.use_mask_attention = config.get("mask_attention", False)
        if self.use_mask_attention:
            self.controller = MaskAttentionController(model)

    def build_messages(self, image: Image.Image, prompt: str) -> list:
        task_prompt = (
            f"Edit only the masked/highlighted region: {prompt}. "
            "Do NOT change anything outside the specified region. "
            "Describe the exact pixel-level changes you would make within the region, "
            "and confirm that the surrounding area is preserved exactly."
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
        Run local editing within the masked region.

        Args:
            image:  Input image
            prompt: e.g. "change the shirt color to red"
            mask:   Binary mask of the edit region (white = edit here)

        Returns:
            (edited_image, edit_description)
        """
        image = self.preprocess_image(image)

        if mask is None:
            raise ValueError(
                "[local_edit] Mask is required for local editing. "
                "Provide a binary PIL image with white = edit region."
            )

        # Mask-guided attention gating is the core technique for this task
        if self.use_mask_attention:
            self.controller.set_mask(mask, image_size=(self.image_size, self.image_size))
            self.controller.enable()

        edit_description = self.generate(image, prompt)

        if self.use_mask_attention:
            self.controller.disable()
            self.controller.clear_hooks()

        edited_image = self._apply_local_edit(image, edit_description, mask)
        return edited_image, edit_description

    def _apply_local_edit(
        self,
        image: Image.Image,
        description: str,
        mask: Image.Image,
    ) -> Image.Image:
        """
        Placeholder for diffusion-based local editing.

        A key design constraint: the output outside the mask must be
        IDENTICAL to the input (measured by PSNR in eval). Make sure
        your diffusion call composites the output with the original
        outside the mask:

            edited = pipe(prompt=description, image=image, mask_image=mask).images[0]
            # Hard-composite: keep original pixels outside mask
            mask_np = np.array(mask.convert("L")) / 255.0
            mask_3ch = np.stack([mask_np]*3, axis=-1)
            final = np.array(edited) * mask_3ch + np.array(image) * (1 - mask_3ch)
            return Image.fromarray(final.astype(np.uint8))
        """
        # TODO: replace with actual diffusion call + hard composite
        print(f"[local_edit] Edit description: {description[:120]}...")

        # Placeholder: tint the masked region to show it was processed
        img_np = np.array(image).astype(np.float32)
        mask_np = np.array(mask.convert("L").resize(image.size)) / 255.0
        mask_3ch = np.stack([mask_np] * 3, axis=-1)

        # Slight yellow tint on masked region for visualization
        tint = np.array([255, 220, 100], dtype=np.float32)
        tinted = img_np * 0.7 + tint * 0.3
        result = img_np * (1 - mask_3ch) + tinted * mask_3ch

        return Image.fromarray(result.clip(0, 255).astype(np.uint8))

    def compute_outside_preservation(
        self,
        original: Image.Image,
        edited: Image.Image,
        mask: Image.Image,
    ) -> float:
        """
        Compute mean squared error of unmasked region.
        This is the key metric for local editing — measures how well
        the model preserved pixels it wasn't supposed to change.

        Lower = better preservation.
        """
        orig_np = np.array(original.convert("RGB")).astype(np.float32)
        edit_np = np.array(edited.convert("RGB")).astype(np.float32)
        mask_np = np.array(mask.convert("L").resize(original.size)) / 255.0

        # Outside mask region
        outside = (1 - mask_np)[:, :, None]
        mse = ((orig_np - edit_np) ** 2 * outside).sum() / (outside.sum() * 3 + 1e-8)
        psnr = 10 * np.log10(255**2 / (mse + 1e-8))
        return float(psnr)
