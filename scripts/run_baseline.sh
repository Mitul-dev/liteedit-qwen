#!/bin/bash
# scripts/run_baseline.sh
# -----------------------
# Run the baseline evaluation on all three tasks.
# Results are saved under results/.
#
# Usage:
#   chmod +x scripts/run_baseline.sh
#   ./scripts/run_baseline.sh
#
# On Kaggle/Colab: paste the python command directly into a cell.

set -e

echo "============================================"
echo " LiteEdit-Qwen — Baseline Evaluation"
echo "============================================"

# Build demo datasets if not yet present
if [ ! -d "data/test/bg_replace" ]; then
    echo "[setup] Building demo datasets..."
    python data/dataset_bg.py     --build_demo --data_dir data --n_images 10
    python data/dataset_remove.py --build_demo --data_dir data --n_images 10
    python data/dataset_local.py  --build_demo --data_dir data --n_images 10
    echo "[setup] Demo datasets ready."
fi

# Verify
echo "[verify] Checking datasets..."
python data/dataset_bg.py     --verify --data_dir data
python data/dataset_remove.py --verify --data_dir data
python data/dataset_local.py  --verify --data_dir data

# Run baseline — outputs go to results/ automatically via PATHS
echo ""
echo "[eval] Running baseline..."
python eval/run_eval.py \
    --configs configs/baseline.yaml \
    --tasks bg_replace person_remove local_edit \
    --data_dir data/test \
    --max_samples 10

echo ""
echo "============================================"
echo " Done. Results saved under results/"
echo "   Images   → results/images/"
echo "   Metrics  → results/metrics/ablation_results.csv"
echo "   Per-img  → results/metrics/*_metrics.json"
echo ""
echo " Next: run scripts/run_ablation.sh"
echo "============================================"
