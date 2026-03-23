"""
models/loader.py
----------------
Loads Qwen2.5-VL with the config specified in a YAML file.
Supports: fp16 baseline, INT4, INT8, and optional LoRA adapter attachment.

Usage:
    from models.loader import load_model
    model, processor = load_model("configs/baseline.yaml")
"""

import yaml
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
import os


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_quant_config(quantization: str):
    """Build BitsAndBytesConfig for INT4 or INT8."""
    if quantization == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 GB
            bnb_4bit_quant_type="nf4",         # nf4 = best quality for LLMs
        )
    elif quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None


def load_model(config_path: str):
    """
    Load model and processor according to a YAML config.

    Returns:
        model: Qwen2VLForConditionalGeneration (possibly quantized + LoRA)
        processor: AutoProcessor
        config: dict (parsed YAML, passed downstream to tasks)
    """
    config = load_config(config_path)

    model_id = config["model_id"]
    quantization = config.get("quantization", None)
    lora_path = config.get("lora_path", None)
    device_map = config.get("device_map", "auto")
    dtype_str = config.get("dtype", "float16")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    quant_config = build_quant_config(quantization)

    print(f"[loader] Loading {model_id}")
    print(f"[loader] Quantization: {quantization or 'none (fp16)'}")
    print(f"[loader] Device map: {device_map}")

    model_kwargs = dict(
        torch_dtype=dtype if quant_config is None else None,
        device_map=device_map,
    )
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)

    # Attach LoRA adapter if specified
    if lora_path and os.path.exists(lora_path):
        print(f"[loader] Attaching LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # merge for faster inference
    elif lora_path:
        print(f"[loader] WARNING: lora_path set but {lora_path} not found. Skipping.")

    processor = AutoProcessor.from_pretrained(model_id)

    # Report VRAM usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        print(f"[loader] VRAM after load: {vram_used:.2f} GB")

    model.eval()
    return model, processor, config

