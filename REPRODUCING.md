# Reproducing the Experiments

These instructions assume Windows PowerShell and a single NVIDIA GPU. The original experiments were run on an NVIDIA GeForce RTX 5060 Ti with 16GB VRAM.

## 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-torch-cu128.txt
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH='src'
.\.venv\Scripts\python scripts\smoke_gpu.py
```

If your GPU or CUDA version differs, install the PyTorch build appropriate for your system.

## 2. Prepare Data

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python scripts\export_pubmedqa_splits.py --out-dir data\pubmedqa
.\.venv\Scripts\python scripts\export_medqa_subset.py --split train --out-dir data\medqa
.\.venv\Scripts\python scripts\export_medqa_subset.py --split test --out-dir data\medqa
.\.venv\Scripts\python scripts\export_medmcqa_subset.py --split train --max-samples 30000 --out-dir data\medmcqa
.\.venv\Scripts\python scripts\export_medmcqa_subset.py --split validation --max-samples 1000 --out-dir data\medmcqa
.\.venv\Scripts\python scripts\build_multitask_sft.py --pubmedqa-repeat 6 --medmcqa-train data\medmcqa\train_30000_seed42.jsonl --medqa-train data\medqa\train_full_seed42.jsonl --out data\multitask\sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42.jsonl
```

The resulting full45k replay mixture contains 44,978 examples.

## 3. Train and Evaluate the Main Matrix

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python scripts\run_full_matrix.py --models qwen05b qwen15b qwen3b --seeds 42 43 44 --skip-existing
```

This trains LoRA adapters and evaluates PubMedQA test100, MedMCQA val1000, and MedQA test1273.

## 4. Build Collapse-Aware Reweighted Training Data

The optional CAW strategy keeps the same examples but adds per-example `sample_weight` values:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python scripts\build_collapse_aware_weights.py --input data\multitask\sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42.jsonl --out data\multitask\sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42_caw_p05.jsonl --group-by source --power 0.5
```

Run the CAW comparison:

```powershell
.\.venv\Scripts\python scripts\run_full_matrix.py --models qwen05b qwen15b qwen3b --seeds 42 43 44 --train-file data\multitask\sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42_caw_p05.jsonl --strategy-tag caw45k_p05 --use-sample-weights --skip-existing
```

## 5. Generate Detailed Diagnostics

Use the benchmarker and diagnostic scripts to regenerate the tables under `results/`.

The manuscript reports:

- accuracy
- macro-F1
- per-class recall/F1
- confusion matrices
- prediction counts
- dominant-label rate
- normalized prediction entropy
- collapse flags
- majority/uniform/prior baselines
- paired McNemar tests
- split/leakage audit

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python scripts\benchmarker.py --root . --out-prefix runs\benchmarker_current
.\.venv\Scripts\python scripts\paired_prediction_stats.py --root . --out runs\paired_stats_current.csv
.\.venv\Scripts\python scripts\detailed_error_tables.py --root . --prediction-index runs\benchmarker_predictions_current.csv --out-dir runs
.\.venv\Scripts\python scripts\data_split_audit.py --root . --out-prefix runs\data_split_audit_current
```

See `ABLATION_MATRIX.md` for replay-ratio, no-resampling, prompt-only/format, single-task, multi-task, and class-balanced ablation commands.

## Notes

The repository intentionally excludes:

- `.venv`
- model checkpoints
- LoRA adapter weights
- full downloaded benchmark datasets

The repository includes raw prediction JSONL files for the main evaluation matrix so reported p-values and per-seed tables can be audited.
