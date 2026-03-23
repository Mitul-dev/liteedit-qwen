# LiteEdit-Qwen

Lightweight VLM-based image editing with Qwen2.5-VL.
Targets three tasks: background replacement, person removal, mask-guided local editing.
Tests four efficiency techniques: INT4 quantization, low-step scheduling, LoRA adapters, mask-guided attention gating.

---

## Quickstart (Kaggle / Colab)

```bash
pip install -r requirements.txt

# Build demo datasets (no download needed)
python data/dataset_bg.py     --build_demo --data_dir data
python data/dataset_remove.py --build_demo --data_dir data
python data/dataset_local.py  --build_demo --data_dir data

# Run baseline — outputs go to results/ automatically
python eval/run_eval.py \
    --configs configs/baseline.yaml \
    --tasks bg_replace person_remove local_edit \
    --data_dir data/test

# Generate LaTeX tables
python eval/ablation_table.py
```

---

## Results layout

All experiment outputs are saved under `results/`:

```
results/
├── images/                  Edited output images
│   └── {config}_{task}/     e.g. baseline_bg_replace/
│       └── img_0000.png
├── metrics/                 Numbers
│   ├── ablation_results.csv           ← master ablation CSV
│   ├── {config}_{task}_metrics.json   ← per-image breakdown
│   ├── step_sweep.csv                 ← from notebook 02
│   ├── lora_impact.csv                ← from notebook 03
│   └── quant_quick.csv                ← from notebook 01
├── tables/                  Paper-ready LaTeX (commit these)
│   ├── ablation_table.tex             ← one table per task
│   └── summary_table.tex              ← cross-task overview
├── figures/                 Paper-ready plots (commit these)
│   ├── quant_comparison.png
│   ├── step_sweep.png
│   └── lora_comparison.png
└── lora_checkpoints/        Trained adapter weights
    ├── bg_replace_adapter/
    ├── person_remove_adapter/
    └── local_edit_adapter/
```

---

## Project structure

```
liteedit-qwen/
├── paths.py            Single source of truth for all project paths
├── configs/            One YAML per ablation condition
├── models/
│   ├── loader.py       Load Qwen with quantization + LoRA
│   ├── lora_wrapper.py Attach/detach adapters at runtime
│   └── mask_attention.py  Novel: mask-guided attention gating hooks
├── tasks/
│   ├── base_task.py    Abstract: run(image, prompt, mask) → image
│   ├── bg_replace.py
│   ├── person_remove.py
│   └── local_edit.py
├── data/
│   ├── dataset_bg.py       Dataset + demo builder
│   ├── dataset_remove.py
│   └── dataset_local.py
├── train/
│   └── train_lora.py   LoRA fine-tuning (PEFT)
├── eval/
│   ├── metrics.py      LPIPS · PSNR · outside-PSNR · FID · latency · VRAM
│   ├── run_eval.py     Ablation runner → results/metrics/ + results/images/
│   └── ablation_table.py  CSV → LaTeX → results/tables/
├── scripts/
│   ├── run_baseline.sh
│   ├── run_ablation.sh
│   ├── train_all_lora.sh
│   └── demo.py         Gradio demo (saves to results/images/demo/)
└── notebooks/
    ├── 01_quant_explore.ipynb  → results/figures/quant_comparison.png
    ├── 02_step_sweep.ipynb     → results/figures/step_sweep.png
    └── 03_lora_finetune.ipynb  → results/figures/lora_comparison.png
```

---

## Ablation configs

| Config | Quant | Steps | LoRA | Mask gating |
|---|---|---|---|---|
| `baseline.yaml`    | fp16 | 50 | ✗ | ✗ |
| `quant_int4.yaml`  | INT4 | 50 | ✗ | ✗ |
| `lowstep.yaml`     | fp16 | 10 | ✗ | ✗ |
| `lora_bg.yaml`     | INT4 | 50 | ✓ bg | ✗ |
| `lora_remove.yaml` | INT4 | 50 | ✓ remove | ✓ |
| `lora_local.yaml`  | INT4 | 50 | ✓ local | ✓ |
| `full_lite.yaml`   | INT4 | 10 | (set at runtime) | ✓ |

---

## Full workflow

```bash
# 1. Build demo data
python data/dataset_bg.py --build_demo --data_dir data --n_images 10
python data/dataset_remove.py --build_demo --data_dir data --n_images 10
python data/dataset_local.py --build_demo --data_dir data --n_images 10

# 2. Baseline (establishes Row 1 of your table)
bash scripts/run_baseline.sh

# 3. Train LoRA adapters (~45 min on RTX 3090)
bash scripts/train_all_lora.sh

# 4. Full ablation sweep
bash scripts/run_ablation.sh
# → results/metrics/ablation_results.csv
# → results/tables/ablation_table.tex
# → results/tables/summary_table.tex

# 5. Gradio demo for paper figures
python scripts/demo.py --config configs/full_lite.yaml
```

---

## Recommended datasets

| Task | Dataset | Link |
|---|---|---|
| Background replace | BG-20k | https://github.com/JizhiziLi/GFM |
| Person removal | INRIA Person | http://pascal.inrialpes.fr/data/human/ |
| Local editing | MagicBrush | https://huggingface.co/datasets/osunlp/MagicBrush |

---

## Novel contribution

`models/mask_attention.py` — **mask-guided attention gating**: a training-free
inference-time hook that suppresses cross-attention weights outside the user mask.
No training required. Biggest impact on `local_edit` and `person_remove` tasks.

---

## Citation

```bibtex
@misc{liteedit-qwen-2025,
  title  = {LiteEdit-Qwen: Lightweight VLM-Based Image Editing},
  author = {Your Name},
  year   = {2025},
}
```
