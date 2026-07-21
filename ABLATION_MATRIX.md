# Ablation Matrix

This file defines the command-level ablations used to test whether the main result depends on the replay mixture, task mixture, prompt format, or class balance. Do not treat an ablation as a manuscript result unless the corresponding per-seed CSV files are present under `results/`.

All commands assume PowerShell from the repository root:

```powershell
$env:PYTHONPATH='src'
```

## Replay Ratios

Build PubMedQA replay variants while keeping MedMCQA and MedQA fixed:

```powershell
.\.venv\Scripts\python scripts\build_multitask_sft.py --pubmedqa-repeat 1 --medmcqa-train data\medmcqa\train_30000_seed42.jsonl --medqa-train data\medqa\train_full_seed42.jsonl --out data\multitask\sft_pubmedqa_x1_medmcqa30k_medqa10k_seed42.jsonl
.\.venv\Scripts\python scripts\build_multitask_sft.py --pubmedqa-repeat 3 --medmcqa-train data\medmcqa\train_30000_seed42.jsonl --medqa-train data\medqa\train_full_seed42.jsonl --out data\multitask\sft_pubmedqa_x3_medmcqa30k_medqa10k_seed42.jsonl
.\.venv\Scripts\python scripts\build_multitask_sft.py --pubmedqa-repeat 6 --medmcqa-train data\medmcqa\train_30000_seed42.jsonl --medqa-train data\medqa\train_full_seed42.jsonl --out data\multitask\sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42.jsonl
```

Train/evaluate each ratio:

```powershell
.\.venv\Scripts\python scripts\run_full_matrix.py --models qwen05b qwen15b qwen3b --seeds 42 43 44 --train-file data\multitask\sft_pubmedqa_x1_medmcqa30k_medqa10k_seed42.jsonl --strategy-tag replay_x1 --skip-existing
.\.venv\Scripts\python scripts\run_full_matrix.py --models qwen05b qwen15b qwen3b --seeds 42 43 44 --train-file data\multitask\sft_pubmedqa_x3_medmcqa30k_medqa10k_seed42.jsonl --strategy-tag replay_x3 --skip-existing
.\.venv\Scripts\python scripts\run_full_matrix.py --models qwen05b qwen15b qwen3b --seeds 42 43 44 --train-file data\multitask\sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42.jsonl --strategy-tag replay_x6 --skip-existing
```

## No PubMedQA Resampling

The `replay_x1` condition above is the no-resampling comparison against the main `replay_x6/full45k` mixture.

## Single-Task Versus Multi-Task

Build task-specific LoRA training files and compare each task to the full45k multi-task mixture:

```powershell
.\.venv\Scripts\python scripts\build_multitask_sft.py --pubmedqa-repeat 1 --max-medmcqa 0 --max-medqa 0 --out data\multitask\sft_pubmedqa_only_seed42.jsonl
.\.venv\Scripts\python scripts\build_multitask_sft.py --pubmedqa-repeat 0 --medmcqa-train data\medmcqa\train_30000_seed42.jsonl --max-medqa 0 --out data\multitask\sft_medmcqa_only_seed42.jsonl
.\.venv\Scripts\python scripts\build_multitask_sft.py --pubmedqa-repeat 0 --max-medmcqa 0 --medqa-train data\medqa\train_full_seed42.jsonl --out data\multitask\sft_medqa_only_seed42.jsonl
```

Run each file through `scripts/run_full_matrix.py` with a unique `--strategy-tag`.

## Prompt-Only / Format-Constrained Baseline

Frozen prompt-only baselines use the same parsing and deterministic generation as the main matrix. For PubMedQA prompt variants:

```powershell
.\.venv\Scripts\python scripts\run_pubmedqa_frozen.py --template-id 0 --out runs\pubmedqa_qwen05b_frozen_template0_test100.jsonl
.\.venv\Scripts\python scripts\run_pubmedqa_frozen.py --template-id 1 --out runs\pubmedqa_qwen05b_frozen_template1_test100.jsonl
.\.venv\Scripts\python scripts\run_pubmedqa_frozen.py --template-id 2 --out runs\pubmedqa_qwen05b_frozen_template2_test100.jsonl
```

Multiple-choice prompt-only baselines use `scripts/run_medmcqa_eval.py` with no adapter path.

## Class-Balanced Sampling / Weighting

Build collapse-aware sample weights and run the weighted trainer:

```powershell
.\.venv\Scripts\python scripts\build_collapse_aware_weights.py --input data\multitask\sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42.jsonl --out data\multitask\sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42_caw_p05.jsonl --group-by source --power 0.5
.\.venv\Scripts\python scripts\run_full_matrix.py --models qwen05b qwen15b qwen3b --seeds 42 43 44 --train-file data\multitask\sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42_caw_p05.jsonl --strategy-tag caw45k_p05 --use-sample-weights --skip-existing
```

After each ablation, regenerate benchmarker, per-class, confusion, prediction-count, split-audit, and paired-test tables using the commands in `REPRODUCING.md`.
