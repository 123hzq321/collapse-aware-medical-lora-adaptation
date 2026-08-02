# Collapse-Aware LoRA Adaptation for Resource-Limited Medical Language Models

This repository contains code, aggregate results, and manuscript source files for:

**Collapse-Aware LoRA Adaptation for Resource-Limited Medical Language Models**

The study evaluates whether lightweight LoRA adaptation can reduce answer collapse in small medical language models under limited compute.

## Public Release

A complete source package, including scripts, manuscript source, result tables, raw predictions, per-class diagnostics, confusion matrices, prediction counts, split audit, and paired-test p-values, is attached to the public `v0.1` release:

https://github.com/123hzq321/collapse-aware-medical-lora-adaptation/releases/tag/v0.1

## Summary

Small language models can obtain non-trivial medical QA accuracy while overusing one answer label. We study this failure mode as **answer collapse** and evaluate Qwen2.5-Instruct models at 0.5B, 1.5B, and 3B parameters on PubMedQA, MedMCQA, and MedQA.

The main full45k LoRA adaptation runs use a 44,978-example multi-task medical QA replay mixture and three random seeds. The benchmarker reports accuracy, macro-F1, per-class recall/F1, confusion matrices, prediction counts, dominant-label rate, normalized prediction entropy, collapse flags, non-model baselines, frozen-model baselines, and paired McNemar tests.

## Main Finding

Across all tested model scales and tasks, full45k LoRA adaptation removed collapse in every seed and improved task performance. Scaling alone did not guarantee stable behavior: frozen 1.5B and 3B models still collapsed on PubMedQA.

## External Generalization Extension

The original PubMedQA test100 slice is useful for iteration but too small for broad generalization claims. This repository now includes an external benchmark manifest and a unified exporter/evaluator for a larger follow-up suite:

- PubMedQA labeled all-example 5-fold evaluation.
- MMLU medical and biomedical subsets: anatomy, clinical knowledge, college biology, college medicine, medical genetics, and professional medicine.
- Optional BioASQ yes/no and MedNLI checks, subject to their official access terms.
- BoolQ and GLUE RTE retention checks for nonmedical generalization.

Generated local benchmark sizes in the working project:

- PubMedQA full labeled pool: 1,000 rows plus five stratified test folds of 201/201/200/199/199 rows.
- MMLU medical/biomedical test: 1,089 rows.
- BoolQ validation: 3,270 rows.
- GLUE RTE validation: 277 rows.

See `EXTERNAL_BENCHMARKS.md`, `configs/external_benchmarks.yml`, `scripts/export_external_benchmarks.py`, and `scripts/run_external_eval.py`.

Important: the PubMedQA full pool is for cross-validation. Do not report it as a clean test set for adapters trained on the original 800-row PubMedQA training split; train one adapter per fold and evaluate on the paired held-out fold.

## Repository Layout

- `src/medself/`: project Python package.
- `scripts/`: data export, training, evaluation, aggregation, and benchmarker scripts.
- `configs/`: small configuration files and benchmark manifests.
- `results/`: aggregate result tables and paper-ready summaries.
- `raw_predictions/`: per-example prediction JSONL files for the main frozen and full45k runs.
- `paper/`: manuscript source files.
- `requirements.txt`: Python dependencies.
- `requirements-torch-cu128.txt`: CUDA 12.8 PyTorch install notes for the local GPU setting.
- `REPRODUCING.md`: step-by-step reproduction instructions.
- `ABLATION_MATRIX.md`: command-level ablation matrix for replay ratios, no-resampling, prompt-only, single-task, multi-task, and class-balanced conditions.
- `EXTERNAL_BENCHMARKS.md`: external generalization benchmark extension.

## Data

This repository does not include full benchmark datasets, downloaded model weights, LoRA adapter weights, or generated training files. It does include raw prediction JSONL files for the main evaluation matrix so that per-seed statistics, confusion matrices, prediction counts, and paired tests can be audited. The scripts download or reconstruct the required data from the original public providers:

- PubMedQA
- MedMCQA
- MedQA-USMLE
- MMLU
- BoolQ
- GLUE RTE
- Qwen2.5-Instruct

BioASQ and MedNLI are listed as optional external validation sets because their raw data should be obtained through their official access routes.

## Clinical Use

This project is a benchmark study. It is not a clinical decision-support system and should not be used for diagnosis, treatment, or patient care.

## Citation

If you use this code or results, please cite the manuscript:

```bibtex
@misc{hu2026collapseaware,
  title = {Collapse-Aware LoRA Adaptation for Resource-Limited Medical Language Models},
  author = {Hu, Zhequan and Bi, Xingang},
  year = {2026},
  note = {Manuscript in preparation}
}
```
