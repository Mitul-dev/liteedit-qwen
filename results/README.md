# results/

All experiment outputs are saved here. Never commit the contents of `images/`
or `metrics/` to git — add them to `.gitignore`. Only commit `tables/` and
`figures/` when you're ready to share paper-ready outputs.

```
results/
├── images/                  Edited output images, organised by experiment
│   └── {config}_{task}/     e.g. baseline_bg_replace/, quant_int4_local_edit/
│       └── img_0000.png
│
├── metrics/                 Raw numbers from every experiment run
│   ├── baseline_results.csv
│   ├── ablation_results.csv
│   └── {config}_{task}_metrics.json   per-image breakdown
│
├── tables/                  Paper-ready LaTeX tables (commit these)
│   ├── ablation_table.tex
│   └── summary_table.tex
│
├── figures/                 Paper-ready plots (commit these)
│   ├── quant_comparison.png
│   ├── step_sweep_full.png
│   └── lora_comparison.png
│
└── lora_checkpoints/        Trained LoRA adapter weights
    ├── bg_adapter/
    ├── remove_adapter/
    └── local_adapter/
```

## Naming convention for experiment images

`results/images/{config_name}_{task_name}/`

Examples:
- `results/images/baseline_bg_replace/`
- `results/images/quant_int4_person_remove/`
- `results/images/full_lite_local_edit/`
