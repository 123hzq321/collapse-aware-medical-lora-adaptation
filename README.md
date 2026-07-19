# Collapse-Aware Self-Update for Resource-Limited Medical Language Models

This repository contains code, aggregate results, and manuscript source files for:

**Collapse-Aware Self-Update for Resource-Limited Medical Language Models**

The study evaluates whether lightweight LoRA adaptation can reduce answer collapse in small medical language models under limited compute.

## Summary

Small language models can obtain non-trivial medical QA accuracy while overusing one answer label. We study this failure mode as **answer collapse** and evaluate Qwen2.5-Instruct models at 0.5B, 1.5B, and 3B parameters on PubMedQA, MedMCQA, and MedQA.

The main full45k LoRA self-update runs use a 44,978-example multi-task medical QA replay mixture and three random seeds. The benchmarker reports accuracy, macro-F1, dominant-label rate, normalized prediction entropy, collapse flags, non-model baselines, frozen-model baselines, and paired McNemar tests.

## Main Finding

Across all tested model scales and tasks, full45k LoRA self-update removed collapse in every seed and improved task performance. Scaling alone did not guarantee stable behavior: frozen 1.5B and 3B models still collapsed on PubMedQA.

## Repository Layout

- `src/medself/`: project Python package.
- `scripts/`: data export, training, evaluation, aggregation, and benchmarker scripts.
- `configs/`: small configuration files.
- `results/`: aggregate result tables and paper-ready summaries.
- `paper/`: manuscript source files.
- `requirements.txt`: Python dependencies.
- `requirements-torch-cu128.txt`: CUDA 12.8 PyTorch install notes for the local GPU setting.
- `REPRODUCING.md`: step-by-step reproduction instructions.

## Data

This repository does not include full benchmark datasets, downloaded model weights, LoRA adapter weights, or generated per-example prediction files. The scripts download or reconstruct the required data from the original public providers:

- PubMedQA
- MedMCQA
- MedQA-USMLE
- Qwen2.5-Instruct

## Clinical Use

This project is a benchmark study. It is not a clinical decision-support system and should not be used for diagnosis, treatment, or patient care.

## Citation

If you use this code or results, please cite the manuscript:

```bibtex
@misc{hu2026collapseaware,
  title = {Collapse-Aware Self-Update for Resource-Limited Medical Language Models},
  author = {Hu, Zhequan and Bi, Xingang},
  year = {2026},
  note = {Manuscript in preparation}
}
```
