"""
tasks/base_task.py
------------------
Abstract base class for all three editing tasks.
All tasks share the same interface: run(image, prompt, mask) -> Image.

This makes the eval loop task-agnostic — it just calls task.run() for any task.
"""

from abc import ABC, abstractmethod
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info


class BaseTask(ABC):

    def __init__(self, model, processor, config: dict):
        self.model = model
        self.processor = processor
        self.config = config
        self.max_new_tokens = config.get("max_new_tokens", 1024)
        self.image_size = config.get("image_size", 512)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Resize image to configured size."""
        return image.convert("RGB").resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )

    def build_messages(self, image: Image.Image, prompt: str) -> list:
        """
        Build the message list in Qwen2.5-VL chat format.
        Override in subclasses to add task-specific system prompts.
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

    def generate(self, image: Image.Image, prompt: str) -> str:
        """Run one forward pass and return the model's text output."""
        image = self.preprocess_image(image)
        messages = self.build_messages(image, prompt)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        # Strip the input tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    @abstractmethod
    def run(self, image: Image.Image, prompt: str, mask: Image.Image = None) -> Image.Image:
        """
        Run the editing task.

        Args:
            image:  Input PIL image
            prompt: Text instruction describing the edit
            mask:   Optional binary mask (white = edit region)

        Returns:
            Edited PIL image
        """
        pass
