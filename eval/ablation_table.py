"""
eval/ablation_table.py
-----------------------
Reads results/metrics/ablation_results.csv
Writes results/tables/ablation_table.tex  (one table per task)
        results/tables/summary_table.tex  (single cross-task overview)

Usage:
    python eval/ablation_table.py
    python eval/ablation_table.py --csv results/metrics/my_results.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from paths import PATHS


CONFIG_LABELS = {
    "baseline":    r"\textbf{Baseline} (fp16, 50-step)",
    "quant_int4":  r"\quad + INT4 quant.",
    "lowstep":     r"\quad + Low-step (10)",
    "lora_bg":     r"\quad + LoRA (bg)",
    "lora_remove": r"\quad + LoRA (remove)",
    "lora_local":  r"\quad + LoRA (local)",
    "full_lite":   r"\textbf{LiteEdit-Qwen} (full)",
}

TASK_LABELS = {
    "bg_replace":    "Background Replacement",
    "person_remove": "Person Removal",
    "local_edit":    "Local Editing",
}

METRICS = ["lpips", "psnr", "outside_psnr", "fid", "latency_s", "peak_vram_gb"]

METRIC_HEADERS = {
    "lpips":         r"LPIPS$\downarrow$",
    "psnr":          r"PSNR$\uparrow$",
    "outside_psnr":  r"Out-PSNR$\uparrow$",
    "fid":           r"FID$\downarrow$",
    "latency_s":     r"Lat.(s)$\downarrow$",
    "peak_vram_gb":  r"VRAM(GB)$\downarrow$",
}

# Cells that are "better when lower" vs "better when higher"
LOWER_IS_BETTER = {"lpips", "fid", "latency_s", "peak_vram_gb"}


def load_summary(csv_path) -> pd.DataFrame:
    df = pd.read_csv(str(csv_path))
    return df[df["filename"] == "MEAN"].copy()


def fmt(value, metric: str) -> str:
    if pd.isna(value) or value is None:
        return "--"
    if metric in ("lpips", "fid"):
        return f"{value:.3f}"
    if metric in ("psnr", "outside_psnr"):
        return f"{value:.2f}"
    if metric == "latency_s":
        return f"{value:.1f}s"
    if metric == "peak_vram_gb":
        return f"{value:.1f}G"
    return str(round(value, 3))


def bold_best(col_values: list, metric: str) -> list:
    """Return the same values with the best one wrapped in \\textbf{}."""
    numerics = []
    for v in col_values:
        try:
            numerics.append(float(str(v).rstrip("sG")))
        except (ValueError, AttributeError):
            numerics.append(None)

    valid = [v for v in numerics if v is not None]
    if not valid:
        return col_values

    best = min(valid) if metric in LOWER_IS_BETTER else max(valid)
    result = []
    for raw, num in zip(col_values, numerics):
        if num is not None and abs(num - best) < 1e-9:
            result.append(r"\textbf{" + raw + "}")
        else:
            result.append(raw)
    return result


def make_task_table(df: pd.DataFrame, task: str) -> str:
    task_df = df[df["task"] == task].copy()
    if task_df.empty:
        return f"% No data for task: {task}\n"

    available = [m for m in METRICS if m in task_df.columns
                 and task_df[m].notna().any()]
    col_fmt = "l" + "r" * len(available)
    header  = " & ".join(
        [r"\textbf{Configuration}"] + [METRIC_HEADERS[m] for m in available]
    )

    # Build formatted cell values per metric column (for bolding)
    col_cells = {m: [] for m in available}
    for _, row in task_df.iterrows():
        for m in available:
            col_cells[m].append(fmt(row.get(m), m))

    # Bold the best value per column
    col_bolded = {m: bold_best(col_cells[m], m) for m in available}

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]

    for i, (_, row) in enumerate(task_df.iterrows()):
        config  = row["config"]
        label   = CONFIG_LABELS.get(config, config)
        cells   = [col_bolded[m][i] for m in available]

        # Horizontal rule before full_lite row
        if config == "full_lite" and i > 0:
            lines.append(r"\midrule")

        lines.append(f"{label} & {' & '.join(cells)} \\\\")

    task_label = TASK_LABELS.get(task, task)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Ablation on \textbf{{{task_label}}}. "
        r"Out-PSNR measures preservation of unedited pixels (local editing only). "
        r"Best value per column in \textbf{bold}.}",
        rf"\label{{tab:ablation_{task}}}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def make_summary_table(df: pd.DataFrame) -> str:
    """Single table showing all tasks × configs on key metrics only."""
    key_metrics = [m for m in ["lpips", "latency_s", "peak_vram_gb"] if m in df.columns]

    # Multi-column header: one block per task
    tasks = [t for t in TASK_LABELS if t in df["task"].unique()]
    n_per_task = len(key_metrics)
    col_fmt = "l" + ("r" * n_per_task + "|") * len(tasks)
    col_fmt = col_fmt.rstrip("|")

    task_header = " & ".join(
        [" "] + [
            r"\multicolumn{" + str(n_per_task) + r"}{c|}{\textbf{" +
            TASK_LABELS[t] + "}}"
            for t in tasks
        ]
    )
    metric_header = " & ".join(
        ["\\textbf{Config}"] + [METRIC_HEADERS[m] for m in key_metrics] * len(tasks)
    )

    configs_ordered = [c for c in CONFIG_LABELS if c in df["config"].unique()]

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{" + col_fmt + "}",
        r"\toprule",
        task_header + r" \\",
        metric_header + r" \\",
        r"\midrule",
    ]

    for config in configs_ordered:
        label = CONFIG_LABELS.get(config, config)
        cells = [label]
        for task in tasks:
            row = df[(df["config"] == config) & (df["task"] == task)]
            for m in key_metrics:
                if row.empty or m not in row.columns:
                    cells.append("--")
                else:
                    cells.append(fmt(row.iloc[0].get(m), m))
        if config == "full_lite":
            lines.append(r"\midrule")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Summary of all ablation configurations across three tasks. "
        r"Each cell shows LPIPS$\downarrow$, latency, and peak VRAM.}",
        r"\label{tab:summary}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Input CSV (default: results/metrics/ablation_results.csv)")
    args = parser.parse_args()

    csv_path = args.csv or PATHS.ablation_csv
    if not os.path.exists(str(csv_path)):
        print(f"[ablation_table] CSV not found: {csv_path}")
        print("Run eval/run_eval.py first to generate results.")
        return

    PATHS.tables.mkdir(parents=True, exist_ok=True)
    df = load_summary(csv_path)

    tasks = [t for t in TASK_LABELS if t in df["task"].unique()]
    all_tables = []

    for task in tasks:
        table = make_task_table(df, task)
        all_tables.append(table)
        print(f"\n--- Per-task table: {task} ---")
        print(table)

    # Write individual task tables
    per_task_path = PATHS.tables / "ablation_table.tex"
    with open(per_task_path, "w") as f:
        f.write("% Generated by eval/ablation_table.py\n")
        f.write("% Requires: \\usepackage{booktabs}\n\n")
        f.write("\n\n".join(all_tables))
    print(f"\n[ablation_table] Per-task tables → {per_task_path}")

    # Write summary table
    summary_table = make_summary_table(df)
    summary_path  = PATHS.tables / "summary_table.tex"
    with open(summary_path, "w") as f:
        f.write("% Generated by eval/ablation_table.py\n")
        f.write("% Requires: \\usepackage{booktabs}\n\n")
        f.write(summary_table)
    print(f"[ablation_table] Summary table   → {summary_path}")

    print(f"\n[ablation_table] All tables saved to: results/tables/")


if __name__ == "__main__":
    main()
