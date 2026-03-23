"""
models/lora_wrapper.py
----------------------
Attach or detach LoRA adapters at inference time without reloading the base model.
This is useful when running the ablation sweep: load base model once,
swap adapters per task.

Usage:
    from models.lora_wrapper import attach_lora, detach_lora

    model = attach_lora(model, "outputs/lora/bg_adapter")
    # ... run inference ...
    model = detach_lora(model)
"""

import os
import torch
from peft import PeftModel, get_peft_model, LoraConfig, TaskType


def attach_lora(model, adapter_path: str):
    """
    Attach a saved LoRA adapter to a base model.
    Returns the model with adapter weights active.
    """
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    print(f"[lora_wrapper] Attaching adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    return model


def detach_lora(model):
    """
    Merge LoRA weights into base model and unload the adapter structure.
    Call this before attaching a different adapter.
    """
    if hasattr(model, "merge_and_unload"):
        print("[lora_wrapper] Merging and unloading LoRA adapter")
        model = model.merge_and_unload()
    else:
        print("[lora_wrapper] Model has no active LoRA adapter to detach.")
    return model


def get_lora_config(rank: int = 8, task_type=TaskType.CAUSAL_LM):
    """
    Returns a standard LoraConfig for Qwen2.5-VL.
    rank=8 is a good default — rank=4 saves memory, rank=16 may improve quality.
    Target modules are the attention projection layers.
    """
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,        # alpha = 2x rank is standard
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
    )


def count_trainable_params(model):
    """Print how many parameters are trainable vs frozen. Use after attaching LoRA."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    print(f"[lora_wrapper] Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
    return trainable, total
