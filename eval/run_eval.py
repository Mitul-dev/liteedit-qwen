"""
eval/run_eval.py
----------------
Main evaluation runner. Iterates over all (config, task) combinations,
runs inference on the test set, and writes results to results/metrics/.

Output structure:
    results/
      metrics/
        ablation_results.csv               ← merged summary CSV
        {config}_{task}_metrics.json       ← per-image breakdown
      images/
        {config}_{task}/
          img_0000.png ...

Usage:
    python eval/run_eval.py --configs configs/baseline.yaml configs/quant_int4.yaml
                            --tasks bg_replace person_remove local_edit
                            --data_dir data/test
"""

import argparse
import csv
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from tqdm import tqdm

from paths import PATHS
from models.loader import load_model
from tasks.bg_replace import BgReplaceTask
from tasks.person_remove import PersonRemoveTask
from tasks.local_edit import LocalEditTask
from eval.metrics import MetricsTracker


TASK_MAP = {
    "bg_replace":    BgReplaceTask,
    "person_remove": PersonRemoveTask,
    "local_edit":    LocalEditTask,
}


def load_test_samples(data_dir: str, task: str, max_samples: int = 20):
    task_dir    = os.path.join(data_dir, task)
    input_dir   = os.path.join(task_dir, "input")
    ref_dir     = os.path.join(task_dir, "ref")
    mask_dir    = os.path.join(task_dir, "mask")
    prompt_file = os.path.join(task_dir, "prompts.txt")

    if not os.path.exists(input_dir):
        print(f"[run_eval] No test data at {input_dir}. Skipping task: {task}")
        return []

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[:max_samples]

    prompts = {}
    if os.path.exists(prompt_file):
        with open(prompt_file) as f:
            for line in f:
                line = line.strip()
                if line and "|" in line:
                    fname, prompt = line.split("|", 1)
                    prompts[fname.strip()] = prompt.strip()

    samples = []
    for fname in image_files:
        samples.append({
            "filename": fname,
            "input":  Image.open(os.path.join(input_dir, fname)).convert("RGB"),
            "ref":    Image.open(os.path.join(ref_dir,  fname)).convert("RGB")
                      if os.path.exists(os.path.join(ref_dir, fname)) else None,
            "mask":   Image.open(os.path.join(mask_dir, fname)).convert("L")
                      if os.path.exists(os.path.join(mask_dir, fname)) else None,
            "prompt": prompts.get(fname, f"Edit this image ({task})"),
        })

    print(f"[run_eval] Loaded {len(samples)} samples for task: {task}")
    return samples


def run_config_task(config_path, task_name, samples, device="cuda"):
    """
    Run one (config, task) pair.
    Saves edited images  → results/images/{config}_{task}/
    Saves per-image JSON → results/metrics/{config}_{task}_metrics.json
    Returns list of CSV rows.
    """
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    print(f"\n{'='*60}")
    print(f"Config: {config_name}  |  Task: {task_name}")
    print(f"{'='*60}")

    img_dir = PATHS.experiment_image_dir(config_name, task_name)
    img_dir.mkdir(parents=True, exist_ok=True)
    PATHS.metrics.mkdir(parents=True, exist_ok=True)

    model, processor, config = load_model(config_path)
    tracker = MetricsTracker(device=device)
    task = TASK_MAP[task_name](model, processor, config)

    csv_rows, per_image_log = [], []
    all_lpips, all_psnr, all_outside_psnr = [], [], []
    all_latencies, all_vrams = [], []

    for sample in tqdm(samples, desc=f"{config_name}/{task_name}"):
        tracker.start_timer()
        edited_image, description = task.run(
            image=sample["input"],
            prompt=sample["prompt"],
            mask=sample["mask"],
        )
        latency, peak_vram = tracker.stop_timer()

        # Save edited image
        save_path = img_dir / sample["filename"]
        edited_image.save(save_path)

        # Compute quality metrics
        metrics = {}
        if sample["ref"] is not None:
            metrics = tracker.compute(
                pred=edited_image,
                ref=sample["ref"],
                original=sample["input"],
                mask=sample["mask"],
            )
            tracker.update_fid(edited_image, sample["ref"])
            all_lpips.append(metrics.get("lpips", 0))
            all_psnr.append(metrics.get("psnr", 0))
            if "outside_psnr" in metrics:
                all_outside_psnr.append(metrics["outside_psnr"])

        all_latencies.append(latency)
        all_vrams.append(peak_vram)

        row = {
            "config":       config_name,
            "task":         task_name,
            "filename":     sample["filename"],
            "image_path":   str(save_path),
            "latency_s":    round(latency, 3),
            "peak_vram_gb": round(peak_vram, 3),
            **{k: round(v, 4) for k, v in metrics.items()},
        }
        csv_rows.append(row)
        per_image_log.append({**row, "prompt": sample["prompt"],
                               "description": description[:300]})

    fid_score = tracker.compute_fid()

    summary = {
        "config":       config_name,
        "task":         task_name,
        "filename":     "MEAN",
        "image_path":   str(img_dir),
        "latency_s":    round(sum(all_latencies) / len(all_latencies), 3),
        "peak_vram_gb": round(max(all_vrams), 3),
        "lpips":        round(sum(all_lpips) / len(all_lpips), 4) if all_lpips else None,
        "psnr":         round(sum(all_psnr)  / len(all_psnr),  4) if all_psnr  else None,
        "outside_psnr": round(sum(all_outside_psnr) / len(all_outside_psnr), 4) if all_outside_psnr else None,
        "fid":          round(fid_score, 2) if fid_score > 0 else None,
    }
    csv_rows.append(summary)

    # Per-image JSON breakdown
    json_path = PATHS.experiment_metrics_json(config_name, task_name)
    with open(json_path, "w") as f:
        json.dump({"config": config_name, "task": task_name,
                   "summary": summary, "per_image": per_image_log}, f, indent=2)

    print(f"[run_eval] Images  → {img_dir}/")
    print(f"[run_eval] Metrics → {json_path}")

    del model
    torch.cuda.empty_cache()
    return csv_rows


def write_csv(rows: list, csv_path):
    if not rows:
        return
    os.makedirs(os.path.dirname(str(csv_path)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run_eval] CSV → {csv_path}")


def print_summary(all_rows):
    mean_rows = [r for r in all_rows if r["filename"] == "MEAN"]
    if not mean_rows:
        return
    print(f"\n{'─'*74}")
    print(f"{'Config':<22} {'Task':<18} {'LPIPS':>7} {'PSNR':>7} {'Lat(s)':>8} {'VRAM':>6}")
    print(f"{'─'*74}")
    for r in mean_rows:
        print(
            f"{r['config']:<22} {r['task']:<18} "
            f"{str(r.get('lpips', '─')):>7} "
            f"{str(r.get('psnr',  '─')):>7} "
            f"{r['latency_s']:>8.2f} "
            f"{r['peak_vram_gb']:>5.1f}G"
        )
    print(f"{'─'*74}")
    print(f"\nAll results saved under:  results/")
    print(f"  Images   → results/images/")
    print(f"  Metrics  → results/metrics/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs",     nargs="+", default=["configs/baseline.yaml"])
    parser.add_argument("--tasks",       nargs="+",
                        default=["bg_replace", "person_remove", "local_edit"])
    parser.add_argument("--data_dir",    default="data/test")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--output_csv",  default=None,
                        help="Override output CSV (default: results/metrics/ablation_results.csv)")
    args = parser.parse_args()

    output_csv = args.output_csv or PATHS.ablation_csv
    all_rows = []

    for config_path in args.configs:
        for task_name in args.tasks:
            samples = load_test_samples(args.data_dir, task_name, args.max_samples)
            if not samples:
                continue
            rows = run_config_task(config_path, task_name, samples, args.device)
            all_rows.extend(rows)

    write_csv(all_rows, output_csv)
    print_summary(all_rows)


if __name__ == "__main__":
    main()
