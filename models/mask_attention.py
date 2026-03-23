"""
models/mask_attention.py
------------------------
Mask-guided attention gating — the novel contribution of LiteEdit-Qwen.

How it works:
  At inference time, we register forward hooks on the cross-attention layers.
  The hook receives the attention weight tensor and suppresses (zeros) attention
  outside the user-provided mask region. This focuses the model's edit capacity
  on the masked region and prevents hallucination in unedited areas.

This requires NO training — it's a pure inference-time modification.

Usage:
    from models.mask_attention import MaskAttentionController

    controller = MaskAttentionController(model)
    controller.set_mask(mask_pil_image, image_size=(512, 512))
    controller.enable()

    output = model.generate(...)   # attention is gated during this call

    controller.disable()
    controller.clear_hooks()
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


class MaskAttentionController:
    """
    Registers forward hooks on Qwen2.5-VL attention layers to gate
    cross-attention outside the provided binary mask.
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.mask_tensor = None
        self.active = False

    def set_mask(self, mask: Image.Image, image_size: tuple = (512, 512)):
        """
        Set the binary mask that defines the edit region.

        Args:
            mask: PIL Image, white = edit region, black = preserve
            image_size: (H, W) matching the input image size
        """
        mask_np = np.array(mask.convert("L").resize(image_size)) / 255.0
        # Binarize: anything > 0.5 is the edit region
        mask_np = (mask_np > 0.5).astype(np.float32)
        self.mask_tensor = torch.from_numpy(mask_np)  # shape: (H, W)
        print(f"[mask_attention] Mask set. Edit region: "
              f"{self.mask_tensor.sum().item():.0f} / "
              f"{self.mask_tensor.numel()} pixels "
              f"({100*self.mask_tensor.mean().item():.1f}%)")

    def _make_hook(self):
        """
        Returns a forward hook function that gates attention weights.
        The hook intercepts the attention output and suppresses weights
        for spatial positions outside the mask.
        """
        def hook(module, input, output):
            if not self.active or self.mask_tensor is None:
                return output

            # output is typically (attn_output, attn_weights) or just attn_output
            # Handle both cases
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output

            # Get device of the attention output
            device = attn_output.device

            # Flatten mask to sequence length and move to device
            # Mask shape: (H, W) -> (H*W,) -> (1, 1, H*W) for broadcasting
            seq_len = attn_output.shape[1]
            mask_flat = self.mask_tensor.view(-1).to(device)

            # If sequence length doesn't match mask size, interpolate
            if mask_flat.shape[0] != seq_len:
                # Interpolate mask to match sequence length
                mask_resized = F.interpolate(
                    self.mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(int(seq_len**0.5), int(seq_len**0.5)),
                    mode="bilinear",
                    align_corners=False
                ).view(-1).to(device)
                # Pad or trim if needed
                if mask_resized.shape[0] < seq_len:
                    mask_resized = F.pad(mask_resized, (0, seq_len - mask_resized.shape[0]), value=1.0)
                else:
                    mask_resized = mask_resized[:seq_len]
                gate = mask_resized.view(1, -1, 1)
            else:
                gate = mask_flat.view(1, -1, 1)

            # Apply gate: multiply attention output by mask
            # Positions outside mask get suppressed toward identity
            gated_output = attn_output * gate + attn_output.detach() * (1 - gate)

            if isinstance(output, tuple):
                return (gated_output,) + output[1:]
            return gated_output

        return hook

    def enable(self):
        """Register hooks on all attention layers and activate gating."""
        if self.mask_tensor is None:
            raise ValueError("Call set_mask() before enable()")

        self.clear_hooks()  # remove any existing hooks first

        hook_fn = self._make_hook()
        hook_count = 0

        for name, module in self.model.named_modules():
            # Target self-attention and cross-attention output projections
            if "attn" in name.lower() and hasattr(module, "forward"):
                h = module.register_forward_hook(hook_fn)
                self.hooks.append(h)
                hook_count += 1

        self.active = True
        print(f"[mask_attention] Enabled. Registered {hook_count} hooks.")

    def disable(self):
        """Deactivate gating without removing hooks."""
        self.active = False
        print("[mask_attention] Disabled.")

    def clear_hooks(self):
        """Remove all registered hooks. Call after inference is done."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        print(f"[mask_attention] Cleared all hooks.")

    def __del__(self):
        self.clear_hooks()
