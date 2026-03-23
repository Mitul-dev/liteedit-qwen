#!/bin/bash
# scripts/run_ablation.sh
# -----------------------
# Full ablation sweep across all configs and all tasks.
# All outputs are saved under results/ via PATHS.
#
# Estimated runtime on RTX 3090 (10 images per task):
#   baseline    ~15 min
#   quant_int4  ~ 8 min
#   lowstep     ~ 4 min
#   full_lite   ~ 3 min
#   Total       ~30–45 min
#
# Set MAX_SAMPLES=50 for paper-quality results.

set -e

MAX_SAMPLES=10  # increase to 50 for final paper numbers

echo "============================================"
echo " LiteEdit-Qwen — Full Ablation Sweep"
echo " MAX_SAMPLES=${MAX_SAMPLES}"
echo "============================================"

# ── Phase 1: configs that don't need LoRA ─────────────────────────
echo ""
echo "[phase 1] Base configs (no LoRA)..."

python eval/run_eval.py \
    --configs \
        configs/baseline.yaml \
        configs/quant_int4.yaml \
        configs/lowstep.yaml \
        configs/full_lite.yaml \
    --tasks bg_replace person_remove local_edit \
    --data_dir data/test \
    --max_samples $MAX_SAMPLES \
    --output_csv results/metrics/ablation_phase1.csv

echo "[phase 1] Done → results/metrics/ablation_phase1.csv"

# ── Phase 2: LoRA configs (only if adapters exist) ────────────────
LORA_CONFIGS=()
[ -d "results/lora_checkpoints/bg_replace_adapter" ]     && LORA_CONFIGS+=("configs/lora_bg.yaml")
[ -d "results/lora_checkpoints/person_remove_adapter" ]  && LORA_CONFIGS+=("configs/lora_remove.yaml")
[ -d "results/lora_checkpoints/local_edit_adapter" ]     && LORA_CONFIGS+=("configs/lora_local.yaml")

if [ ${#LORA_CONFIGS[@]} -gt 0 ]; then
    echo ""
    echo "[phase 2] LoRA configs: ${LORA_CONFIGS[*]}"
    python eval/run_eval.py \
        --configs "${LORA_CONFIGS[@]}" \
        --tasks bg_replace person_remove local_edit \
        --data_dir data/test \
        --max_samples $MAX_SAMPLES \
        --output_csv results/metrics/ablation_phase2.csv
    echo "[phase 2] Done → results/metrics/ablation_phase2.csv"
else
    echo ""
    echo "[phase 2] No LoRA adapters found in results/lora_checkpoints/."
    echo "          Train them first with: bash scripts/train_all_lora.sh"
fi

# ── Merge all phase CSVs into one master CSV ──────────────────────
echo ""
echo "[merge] Merging result CSVs..."
python - <<'PYEOF'
import pandas as pd, glob, os

csvs = sorted(glob.glob("results/metrics/ablation_phase*.csv"))
if not csvs:
    print("  No phase CSVs found.")
else:
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    # Remove duplicate MEAN rows (same config+task) keeping last
    df = df.drop_duplicates(subset=["config", "task", "filename"], keep="last")
    out = "results/metrics/ablation_results.csv"
    df.to_csv(out, index=False)
    print(f"  Merged {len(csvs)} CSVs → {out}  ({len(df)} rows)")
PYEOF

# ── Generate LaTeX tables ─────────────────────────────────────────
echo ""
echo "[tables] Generating LaTeX tables..."
python eval/ablation_table.py
# Outputs:
#   results/tables/ablation_table.tex   (one table per task)
#   results/tables/summary_table.tex    (cross-task overview)

echo ""
echo "============================================"
echo " All done. Results layout:"
echo ""
echo "   results/"
echo "   ├── images/          edited images per experiment"
echo "   ├── metrics/         CSVs and per-image JSON"
echo "   ├── tables/          LaTeX tables for paper"
echo "   └── figures/         plots (run notebooks to populate)"
echo "============================================"
