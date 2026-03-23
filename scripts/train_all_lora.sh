#!/bin/bash
# scripts/train_all_lora.sh
# --------------------------
# Train LoRA adapters for all three tasks.
# Adapters are saved to results/lora_checkpoints/{task}_adapter/
#
# Estimated time on RTX 3090:
#   3 epochs × ~50 samples ≈ 10–15 min per task (~45 min total)
#
# Memory: INT4 base + rank-8 LoRA fits comfortably in 16 GB VRAM.

set -e

BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
RANK=8
EPOCHS=3
LR=2e-4

echo "============================================"
echo " LiteEdit-Qwen — LoRA Training"
echo " Model:  $BASE_MODEL"
echo " Rank:   $RANK | Epochs: $EPOCHS | LR: $LR"
echo " Output: results/lora_checkpoints/"
echo "============================================"

# Build demo training data if not already present
if [ ! -d "data/train/bg_replace" ]; then
    echo "[setup] Building demo training data (50 images per task)..."
    python data/dataset_bg.py     --build_demo --data_dir data --n_images 50
    python data/dataset_remove.py --build_demo --data_dir data --n_images 50
    python data/dataset_local.py  --build_demo --data_dir data --n_images 50
    echo "[setup] Done."
fi

# ── Task 1: Background replacement ───────────────────────────────
echo ""
echo "[1/3] Training bg_replace adapter..."
python train/train_lora.py \
    --task bg_replace \
    --data_dir data/train/bg_replace \
    --base_model $BASE_MODEL \
    --quantize \
    --rank $RANK \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size 1
# output_dir defaults to results/lora_checkpoints/bg_replace_adapter

# ── Task 2: Person removal ────────────────────────────────────────
echo ""
echo "[2/3] Training person_remove adapter..."
python train/train_lora.py \
    --task person_remove \
    --data_dir data/train/person_remove \
    --base_model $BASE_MODEL \
    --quantize \
    --rank $RANK \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size 1

# ── Task 3: Local editing ─────────────────────────────────────────
echo ""
echo "[3/3] Training local_edit adapter..."
python train/train_lora.py \
    --task local_edit \
    --data_dir data/train/local_edit \
    --base_model $BASE_MODEL \
    --quantize \
    --rank $RANK \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size 1

# ── Print adapter sizes ───────────────────────────────────────────
echo ""
echo "============================================"
echo " Adapter sizes:"
for task in bg_replace person_remove local_edit; do
    dir="results/lora_checkpoints/${task}_adapter"
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        loss=$(python -c "
import json, os
p = os.path.join('$dir', 'train_meta.json')
if os.path.exists(p):
    m = json.load(open(p))
    print(f'loss={m.get(\"final_loss\", \"?\")}')
" 2>/dev/null)
        echo "   ${task}: ${size}  ${loss}"
    fi
done

echo ""
echo " Next: bash scripts/run_ablation.sh"
echo "============================================"
