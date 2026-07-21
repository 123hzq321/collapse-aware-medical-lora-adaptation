# BENCHMARKER Current Report

Scope: PubMedQA test100, MedMCQA val1000, and MedQA test1273. Metrics are accuracy, macro-F1, dominant-label rate, normalized prediction entropy, and collapse flag.

## Non-Model Baselines

| task | baseline | condition | accuracy | macro-F1 | expected accuracy | collapse |
|---|---|---|---:|---:|---:|---|
| PubMedQA test100 | gold_majority | always_yes | 0.5600 | 0.2393 |  | True |
| PubMedQA test100 | uniform_random | expected |  |  | 0.3333 |  |
| PubMedQA test100 | label_prior_random | expected |  |  | 0.4392 |  |
| MedMCQA val1000 | gold_majority | always_A | 0.3230 | 0.1221 |  | True |
| MedMCQA val1000 | uniform_random | expected |  |  | 0.2500 |  |
| MedMCQA val1000 | label_prior_random | expected |  |  | 0.2601 |  |
| MedQA test1273 | gold_majority | always_A | 0.2773 | 0.1085 |  | True |
| MedQA test1273 | uniform_random | expected |  |  | 0.2500 |  |
| MedQA test1273 | label_prior_random | expected |  |  | 0.2530 |  |

## Best Current LoRA Adaptation Runs

| task | model | condition | seeds/runs | accuracy | macro-F1 | dominant rate | entropy | collapse |
|---|---|---|---:|---:|---:|---:|---:|---:|
| PubMedQA test100 | qwen15b | full45k | 42 43 44 | 0.6900 | 0.5842 | 0.5433 | 0.8973 | 0/3 |
| MedMCQA val1000 | qwen3b | full45k | 42 43 44 | 0.5833 | 0.5762 | 0.3510 | 0.9723 | 0/3 |
| MedQA test1273 | qwen3b | full45k | 42 43 44 | 0.5656 | 0.5623 | 0.2799 | 0.9951 | 0/3 |

## Frozen vs Full45k Summary

| task | model | frozen acc | full45k acc | delta | frozen collapse | full45k collapse |
|---|---|---:|---:|---:|---:|---:|
| PubMedQA test100 | qwen05b | 0.2100 | 0.6500 | 0.4400 | 1/1 | 0/3 |
| PubMedQA test100 | qwen15b | 0.6200 | 0.6900 | 0.0700 | 1/1 | 0/3 |
| PubMedQA test100 | qwen3b | 0.2300 | 0.6433 | 0.4133 | 1/1 | 0/3 |
| MedMCQA val1000 | qwen05b | 0.3170 | 0.4140 | 0.0970 | 1/1 | 0/3 |
| MedMCQA val1000 | qwen15b | 0.4360 | 0.5390 | 0.1030 | 0/1 | 0/3 |
| MedMCQA val1000 | qwen3b | 0.4900 | 0.5833 | 0.0933 | 0/1 | 0/3 |
| MedQA test1273 | qwen05b | 0.2875 | 0.3888 | 0.1013 | 1/1 | 0/3 |
| MedQA test1273 | qwen15b | 0.4698 | 0.5350 | 0.0652 | 0/1 | 0/3 |
| MedQA test1273 | qwen3b | 0.4965 | 0.5656 | 0.0691 | 0/1 | 0/3 |

## Artifact Index

- `runs/benchmarker_baselines_current.csv`: non-model baselines.
- `runs/benchmarker_predictions_current.csv`: per-run model metrics.
- `runs/benchmarker_aggregate_current.csv`: grouped model metrics.
- `runs/benchmarker_current.md`: this report.
