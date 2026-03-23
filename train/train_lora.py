"""
train/train_lora.py
--------------------
LoRA fine-tuning script for task-specific adapters.

Trains a small rank-8 LoRA adapter on top of the frozen Qwen2.5-VL base model.
One adapter per task — bg_replace, person_remove, local_edit.

Usage:
    python train/train_lora.py \
        --task bg_replace \
        --data_dir data/train/bg_replace \
        --output_dir outputs/lora/bg_adapter \
        --base_model Qwen/Qwen2.5-VL-7B-Instruct \
        --quantize \
        --epochs 3 \
        --rank 8

Memory tip: with INT4 base + rank-8 LoRA, this fits in ~8 GB VRAM.
On a 24 GB GPU (RTX 3090), use rank=16 for better quality.
"""

import argparse
import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import PATHS
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class EditPairDataset(Dataset):
    """
    Loads (input_image, prompt, target_text) triplets for LoRA training.

    Folder structure expected:
        data/train/{task}/
            input/      ← original images
            prompts.txt ← filename|prompt|target_description (one per line)

    The model is trained to generate target_description given
    (input_image, prompt). This teaches it to produce better edit
    descriptions for the specific task.
    """

    def __init__(self, data_dir: str, processor, image_size: int = 512):
        self.data_dir = data_dir
        self.processor = processor
        self.image_size = image_size
        self.samples = self._load_samples()

    def _load_samples(self):
        prompt_file = os.path.join(self.data_dir, "prompts.txt")
        input_dir   = os.path.join(self.data_dir, "input")

        if not os.path.exists(prompt_file):
            raise FileNotFoundError(
                f"prompts.txt not found at {prompt_file}\n"
                f"Format: filename|prompt|target_description"
            )

        samples = []
        with open(prompt_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) < 3:
                    print(f"[dataset] Skipping malformed line: {line[:60]}")
                    continue
                fname, prompt, target = parts[0].strip(), parts[1].strip(), parts[2].strip()
                img_path = os.path.join(input_dir, fname)
                if os.path.exists(img_path):
                    samples.append({
                        "image_path": img_path,
                        "prompt": prompt,
                        "target": target,
                    })
                else:
                    print(f"[dataset] Image not found, skipping: {img_path}")

        print(f"[dataset] Loaded {len(samples)} training samples from {self.data_dir}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB").resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )
        return {
            "image":  image,
            "prompt": sample["prompt"],
            "target": sample["target"],
        }


def collate_fn(batch, processor):
    """Build model inputs from a batch of (image, prompt, target) triplets."""
    messages_list = []
    for item in batch:
        messages_list.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]},
                    {"type": "text",  "text": item["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": item["target"]}],
            },
        ])

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in messages_list
    ]

    image_inputs_list = [process_vision_info(m)[0] for m in messages_list]
    # Flatten images list
    all_image_inputs = [img for sublist in image_inputs_list for img in (sublist or [])]

    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def build_lora_model(base_model_id: str, rank: int, quantize: bool):
    """Load base model (optionally quantized) and attach LoRA config."""

    quant_config = None
    if quantize:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print(f"[train] Loading INT4 quantized base model: {base_model_id}")
    else:
        print(f"[train] Loading fp16 base model: {base_model_id}")

    model_kwargs = dict(device_map="auto")
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id, **model_kwargs
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(args):
    # Default output dir: results/lora_checkpoints/{task}_adapter
    if args.output_dir is None:
        args.output_dir = str(PATHS.lora_adapter_dir(args.task))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[train] Saving adapter to: {args.output_dir}')

    processor = AutoProcessor.from_pretrained(args.base_model)
    model = build_lora_model(args.base_model, args.rank, args.quantize)

    dataset = EditPairDataset(args.data_dir, processor, image_size=args.image_size)

    def _collate(batch):
        return collate_fn(batch, processor)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,   # keep 0 for Windows/Kaggle compatibility
    )

    # Optimizer — only update LoRA params
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = len(dataloader) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n[train] Task: {args.task}")
    print(f"[train] Samples: {len(dataset)}, Epochs: {args.epochs}, "
          f"Batch: {args.batch_size}, LR: {args.lr}, Rank: {args.rank}")
    print(f"[train] Total steps: {total_steps}, Warmup: {warmup_steps}\n")

    model.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()

            # Gradient clipping prevents training instability
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Log every 50 steps
            if global_step % 50 == 0:
                avg_loss = epoch_loss / (pbar.n + 1)
                print(f"  Step {global_step}: loss={avg_loss:.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n[train] Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")

        # Save best checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            ckpt_path = os.path.join(args.output_dir, "best_checkpoint")
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            print(f"[train] New best checkpoint saved to {ckpt_path}")

    # Save final adapter
    final_path = args.output_dir
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    # Save training metadata for reproducibility
    meta = {
        "task":       args.task,
        "base_model": args.base_model,
        "rank":       args.rank,
        "epochs":     args.epochs,
        "lr":         args.lr,
        "batch_size": args.batch_size,
        "quantized":  args.quantize,
        "final_loss": round(best_loss, 4),
        "samples":    len(dataset),
    }
    with open(os.path.join(args.output_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[train] Done. Adapter saved to: {args.output_dir}")
    print(f"[train] Best loss: {best_loss:.4f}")
    print(f"[train] To use this adapter, set lora_path: {args.output_dir} in your config YAML.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train task-specific LoRA adapter")
    parser.add_argument("--task",        required=True,
                        choices=["bg_replace", "person_remove", "local_edit"])
    parser.add_argument("--data_dir",    required=True,
                        help="Path to task training data (must contain input/ and prompts.txt)")
    parser.add_argument("--output_dir",  default=None,
                        help="Where to save adapter weights "
                             "(default: results/lora_checkpoints/{task}_adapter)")
    parser.add_argument("--base_model",  default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--quantize",    action="store_true",
                        help="Use INT4 quantization for base model (saves ~8 GB VRAM)")
    parser.add_argument("--rank",        type=int, default=8,
                        help="LoRA rank. 4=smallest, 8=default, 16=best quality")
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch_size",  type=int, default=1,
                        help="Keep at 1 for 24GB GPU with INT4. Use 2 only on A100 40GB")
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--image_size",  type=int, default=512)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
