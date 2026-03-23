"""
tasks/person_remove.py
-----------------------
Person removal task.

Strategy:
  - Uses mask to identify person region
  - Asks Qwen to describe what should fill the removed area (scene context)
  - Mask-guided attention gating is especially important here:
    suppresses attention on non-person regions during the fill description

Ablation notes:
  - Mask gating matters most for this task — shows biggest delta vs baseline
  - Low-step degrades faster than bg_replace (harder inpainting problem)
  - LoRA should improve coherence of fill (scene continuation)
"""

from PIL import Image
import numpy as np
from tasks.base_task import BaseTask
from models.mask_attention import MaskAttentionController


SYSTEM_PROMPT = (
    "You are an image editing assistant specializing in object removal. "
    "When asked to remove a person or object, describe what the scene should "
    "look like after removal — focusing on what background content would naturally "
    "fill the space. Be specific about textures, colors, and spatial context. "
    "The goal is seamless inpainting with no ghosting or artifacts."
)


class PersonRemoveTask(BaseTask):

    def __init__(self, model, processor, config: dict):
        super().__init__(model, processor, config)
        self.use_mask_attention = config.get("mask_attention", False)
        if self.use_mask_attention:
            self.controller = MaskAttentionController(model)

    def build_messages(self, image: Image.Image, prompt: str) -> list:
        task_prompt = (
            f"{prompt}. "
            "Describe in detail what the scene should look like after the person "
            "is removed — specifically what background content should fill the area "
            "where the person was standing. Include texture, lighting, and color details."
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
        Run person removal.

        Args:
            image:  Input image
            prompt: e.g. "remove the person standing on the left"
            mask:   Binary mask of the person region (white = person)

        Returns:
            (edited_image, fill_description)
        """
        image = self.preprocess_image(image)

        if mask is None:
            print("[person_remove] WARNING: No mask provided. Results may be poor.")
            print("[person_remove] Tip: use SAM or a segmentation model to auto-generate masks.")

        # Enable mask-guided attention gating
        if self.use_mask_attention and mask is not None:
            self.controller.set_mask(mask, image_size=(self.image_size, self.image_size))
            self.controller.enable()

        fill_description = self.generate(image, prompt)

        if self.use_mask_attention and mask is not None:
            self.controller.disable()
            self.controller.clear_hooks()

        edited_image = self._apply_inpaint(image, fill_description, mask)
        return edited_image, fill_description

    def _apply_inpaint(
        self,
        image: Image.Image,
        description: str,
        mask: Image.Image = None,
    ) -> Image.Image:
        """
        Placeholder for diffusion inpainting.
        Replace with actual inpainting pipeline call.

        Example integration:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
            )
            result = pipe(
                prompt=description,
                image=image,
                mask_image=mask,
                num_inference_steps=self.config.get("num_steps", 50),
            )
            return result.images[0]
        """
        # TODO: replace with actual inpainting call
        print(f"[person_remove] Fill description: {description[:120]}...")

        # Naive placeholder: grey out the masked region
        if mask is not None:
            img_np = np.array(image)
            mask_np = np.array(mask.convert("L").resize(image.size)) / 255.0
            mask_3ch = np.stack([mask_np] * 3, axis=-1)
            grey = np.ones_like(img_np) * 128
            blended = img_np * (1 - mask_3ch) + grey * mask_3ch
            return Image.fromarray(blended.astype(np.uint8))

        return image
