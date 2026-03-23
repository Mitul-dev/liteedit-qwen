"""
scripts/demo.py
---------------
Interactive Gradio demo for LiteEdit-Qwen.
Use this to generate paper figures and qualitative examples.

Edited images are saved to results/images/demo/ automatically.

Usage:
    python scripts/demo.py --config configs/full_lite.yaml
    python scripts/demo.py --config configs/baseline.yaml --port 7861
    python scripts/demo.py --share      # public URL (Kaggle/Colab)
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gradio as gr
from PIL import Image

from paths import PATHS
from models.loader import load_model
from tasks.bg_replace import BgReplaceTask
from tasks.person_remove import PersonRemoveTask
from tasks.local_edit import LocalEditTask


# ------------------------------------------------------------------
# Global model (loaded once, reused across all Gradio calls)
# ------------------------------------------------------------------
MODEL      = None
PROCESSOR  = None
CONFIG     = None
TASKS      = {}
DEMO_DIR   = PATHS.images / "demo"


def initialize(config_path: str):
    global MODEL, PROCESSOR, CONFIG, TASKS
    MODEL, PROCESSOR, CONFIG = load_model(config_path)
    TASKS = {
        "bg_replace":    BgReplaceTask(MODEL,    PROCESSOR, CONFIG),
        "person_remove": PersonRemoveTask(MODEL, PROCESSOR, CONFIG),
        "local_edit":    LocalEditTask(MODEL,    PROCESSOR, CONFIG),
    }
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[demo] Model ready.  Config: {config_path}")
    print(f"[demo] Outputs saved to: {DEMO_DIR}")


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def run_edit(task_name, input_image, prompt, mask_image):
    if MODEL is None:
        return None, "Model not loaded."
    if input_image is None:
        return None, "Please upload an input image."
    if not prompt.strip():
        return None, "Please enter an edit prompt."

    input_pil = Image.fromarray(input_image).convert("RGB")
    mask_pil  = Image.fromarray(mask_image).convert("L") if mask_image is not None else None

    task = TASKS.get(task_name)
    if task is None:
        return None, f"Unknown task: {task_name}"

    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        output_image, description = task.run(
            image=input_pil, prompt=prompt, mask=mask_pil
        )
    except Exception as e:
        return None, f"Error: {e}"

    elapsed   = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # Auto-save output to results/images/demo/
    ts = int(time.time())
    save_path = DEMO_DIR / f"{task_name}_{ts}.png"
    output_image.save(save_path)

    info = "\n".join([
        f"**Task:** {task_name}",
        f"**Prompt:** {prompt}",
        f"**Latency:** {elapsed:.2f}s  |  **Peak VRAM:** {peak_vram:.2f} GB",
        f"**Saved to:** `{save_path}`",
        "",
        "**Edit description:**",
        description[:500] + ("..." if len(description) > 500 else ""),
    ])
    return output_image, info


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="LiteEdit-Qwen") as demo:
        gr.Markdown(
            "# LiteEdit-Qwen\n"
            "Lightweight VLM-based image editing — background replacement, "
            "person removal, and mask-guided local editing.\n\n"
            f"Outputs auto-saved to `results/images/demo/`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                task_radio = gr.Radio(
                    choices=["bg_replace", "person_remove", "local_edit"],
                    value="bg_replace",
                    label="Task",
                )
                prompt_box = gr.Textbox(
                    label="Edit prompt",
                    placeholder="e.g. Replace the background with a sunset",
                    lines=2,
                )
                input_image = gr.Image(label="Input image",  type="numpy")
                mask_image  = gr.Image(label="Mask (white = edit region, optional)",
                                       type="numpy")
                run_btn = gr.Button("Run edit ▶", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Output", type="pil")
                info_box     = gr.Markdown()

        gr.Markdown("### Quick examples")
        gr.Examples(
            examples=[
                ["bg_replace",    None, "Replace the background with a sunny beach",   None],
                ["person_remove", None, "Remove the person from the scene",             None],
                ["local_edit",    None, "Change the shirt color to red",                None],
            ],
            inputs=[task_radio, input_image, prompt_box, mask_image],
        )

        run_btn.click(
            fn=run_edit,
            inputs=[task_radio, input_image, prompt_box, mask_image],
            outputs=[output_image, info_box],
        )

    return demo


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true",
                        help="Create public Gradio link (Kaggle/Colab)")
    args = parser.parse_args()

    initialize(args.config)
    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
