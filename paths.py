"""
paths.py
--------
Single source of truth for all project paths.
Import this everywhere instead of hardcoding strings.

Usage:
    from paths import PATHS
    model_output = PATHS.images / "baseline_bg_replace" / "img_0000.png"
    PATHS.metrics.mkdir(parents=True, exist_ok=True)
"""

from pathlib import Path

# Project root = directory containing this file
ROOT = Path(__file__).parent.resolve()

class _Paths:
    # ── Source directories ─────────────────────────────────────────
    configs  = ROOT / "configs"
    models   = ROOT / "models"
    tasks    = ROOT / "tasks"
    train    = ROOT / "train"
    eval     = ROOT / "eval"
    data     = ROOT / "data"
    scripts  = ROOT / "scripts"
    notebooks = ROOT / "notebooks"

    # ── Results (all experiment outputs go here) ───────────────────
    results           = ROOT / "results"
    images            = ROOT / "results" / "images"      # edited images per experiment
    metrics           = ROOT / "results" / "metrics"     # CSVs and JSON metric files
    tables            = ROOT / "results" / "tables"      # LaTeX tables for paper
    figures           = ROOT / "results" / "figures"     # plots and comparison images
    lora_checkpoints  = ROOT / "results" / "lora_checkpoints"  # adapter weights

    # ── Convenience: named result files ───────────────────────────
    @property
    def baseline_csv(self):
        return self.metrics / "baseline_results.csv"

    @property
    def ablation_csv(self):
        return self.metrics / "ablation_results.csv"

    @property
    def ablation_tex(self):
        return self.tables / "ablation_table.tex"

    def experiment_image_dir(self, config_name: str, task_name: str) -> Path:
        """Returns results/images/{config_name}_{task_name}/"""
        return self.images / f"{config_name}_{task_name}"

    def experiment_metrics_json(self, config_name: str, task_name: str) -> Path:
        """Returns results/metrics/{config_name}_{task_name}_metrics.json"""
        return self.metrics / f"{config_name}_{task_name}_metrics.json"

    def lora_adapter_dir(self, task_name: str) -> Path:
        """Returns results/lora_checkpoints/{task_name}_adapter/"""
        return self.lora_checkpoints / f"{task_name}_adapter"

    def ensure_all(self):
        """Create all result directories if they don't exist."""
        for d in [
            self.results, self.images, self.metrics,
            self.tables, self.figures, self.lora_checkpoints,
        ]:
            d.mkdir(parents=True, exist_ok=True)

PATHS = _Paths()

# Auto-create results directories when this module is imported
PATHS.ensure_all()
