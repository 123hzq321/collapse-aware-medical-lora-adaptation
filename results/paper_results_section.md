# Paper-Ready Results Draft

## Chinese Results Draft

我们首先构建了非模型标签基线。PubMedQA 的 majority baseline 为 always-yes，accuracy 为 0.560，但 macro-F1 仅为 0.239，且被 collapse 指标标记为坍缩；MedMCQA 与 MedQA 的 majority baseline 分别为 always-A，accuracy 为 0.323 和 0.277，macro-F1 分别为 0.122 和 0.109，同样属于单标签坍缩。因此，单独报告 accuracy 会高估标签偏置方法的有效性，后续实验同时报告 macro-F1、dominant-rate、entropy 和 collapse flag。

在 0.5B 模型上，full45k self-update 在三个 seed 上稳定改善三项医学任务。PubMedQA 从 frozen baseline 的 0.210 提升到 0.650 +/- 0.046，并将 collapse 从 1/1 降到 0/3。MedMCQA 和 MedQA 也分别获得稳定提升，说明 full45k 不只是改变 dominant label，而是改善了多类覆盖。

在更大模型上，1.5B 与 3B 的 frozen baseline 并不天然免疫 collapse。尤其 3B frozen baseline 在 PubMedQA 上仍出现 maybe-dominant collapse。full45k 后，3B 在 MedMCQA 达到 0.583 +/- 0.003 accuracy / 0.576 +/- 0.001 macro-F1，在 MedQA 达到 0.566 +/- 0.002 accuracy / 0.562 +/- 0.002 macro-F1；PubMedQA 的最佳结果来自 1.5B full45k，达到 0.690 +/- 0.010 accuracy / 0.584 +/- 0.040 macro-F1。

McNemar paired tests 进一步支持这些改善并非简单波动。0.5B full45k 在三任务三 seed 上均显著改善；1.5B full45k 在 MedMCQA 和 MedQA 的三个 seed 上均显著改善，PubMedQA 则稳定解除 collapse 但 accuracy 的配对检验尚未达到显著；3B full45k 在三任务三 seed 上也均达到显著提升，同时解除 PubMedQA collapse。

## English Results Draft

We first report non-model label baselines. The majority baseline reaches 0.560 accuracy on PubMedQA by always predicting yes, but its macro-F1 is only 0.239 and it is flagged as collapsed. Similarly, majority baselines on MedMCQA and MedQA obtain 0.323 and 0.277 accuracy, with macro-F1 scores of 0.122 and 0.109. These results show that accuracy alone can overstate label-biased behavior; we therefore report macro-F1, dominant-label rate, normalized prediction entropy, and collapse flags throughout.

For the 0.5B model, full45k self-update consistently improves all three medical QA tasks across three seeds. On PubMedQA, accuracy improves from 0.210 to 0.650 +/- 0.046, while collapse is reduced from 1/1 to 0/3. MedMCQA and MedQA also improve, indicating that the update improves multi-class coverage rather than merely shifting the dominant label.

Larger models are not automatically immune to collapse. In particular, the 3B frozen baseline still collapses on PubMedQA with a maybe-dominant prediction pattern. After full45k self-update, the 3B model reaches 0.583 +/- 0.003 accuracy / 0.576 +/- 0.001 macro-F1 on MedMCQA and 0.566 +/- 0.002 accuracy / 0.562 +/- 0.002 macro-F1 on MedQA. The best PubMedQA score is obtained by the 1.5B full45k model, with 0.690 +/- 0.010 accuracy / 0.584 +/- 0.040 macro-F1.

Paired McNemar tests support that these gains are not only aggregate fluctuations. The 0.5B full45k model significantly improves over its frozen baseline across all three tasks and seeds. For the 1.5B model, MedMCQA and MedQA improve significantly across all three seeds, while PubMedQA consistently removes collapse without reaching paired-test significance for accuracy. The 3B full45k model also significantly improves over the 3B frozen baseline across all three tasks and seeds, while removing PubMedQA collapse.

## Main Result Table

| Task | Model | Frozen Acc. | Full45k Acc. | Delta Acc. | Full45k Macro-F1 | Frozen Collapse | Full45k Collapse |
|---|---|---|---|---|---|---|---|
| PubMedQA test100 | Qwen2.5-0.5B | 0.210 | 0.650 +/- 0.046 | +0.440 | 0.498 +/- 0.063 | 1/1 | 0/3 |
| PubMedQA test100 | Qwen2.5-1.5B | 0.620 | 0.690 +/- 0.010 | +0.070 | 0.584 +/- 0.040 | 1/1 | 0/3 |
| PubMedQA test100 | Qwen2.5-3B | 0.230 | 0.643 +/- 0.031 | +0.413 | 0.571 +/- 0.045 | 1/1 | 0/3 |
| MedMCQA val1000 | Qwen2.5-0.5B | 0.317 | 0.414 +/- 0.029 | +0.097 | 0.397 +/- 0.030 | 1/1 | 0/3 |
| MedMCQA val1000 | Qwen2.5-1.5B | 0.436 | 0.539 +/- 0.012 | +0.103 | 0.528 +/- 0.011 | 0/1 | 0/3 |
| MedMCQA val1000 | Qwen2.5-3B | 0.490 | 0.583 +/- 0.003 | +0.093 | 0.576 +/- 0.001 | 0/1 | 0/3 |
| MedQA test1273 | Qwen2.5-0.5B | 0.288 | 0.389 +/- 0.023 | +0.101 | 0.381 +/- 0.024 | 1/1 | 0/3 |
| MedQA test1273 | Qwen2.5-1.5B | 0.470 | 0.535 +/- 0.010 | +0.065 | 0.531 +/- 0.010 | 0/1 | 0/3 |
| MedQA test1273 | Qwen2.5-3B | 0.496 | 0.566 +/- 0.002 | +0.069 | 0.562 +/- 0.002 | 0/1 | 0/3 |

## Non-Model Baseline Table

| Task | Baseline | Condition | Accuracy | Macro-F1 | Collapse |
|---|---|---|---|---|---|
| PubMedQA test100 | gold_majority | always_yes | 0.5600 | 0.2393 | True |
| PubMedQA test100 | uniform_random | expected | 0.3333 expected |  |  |
| PubMedQA test100 | label_prior_random | expected | 0.4392 expected |  |  |
| MedMCQA val1000 | gold_majority | always_A | 0.3230 | 0.1221 | True |
| MedMCQA val1000 | uniform_random | expected | 0.2500 expected |  |  |
| MedMCQA val1000 | label_prior_random | expected | 0.2601 expected |  |  |
| MedQA test1273 | gold_majority | always_A | 0.2773 | 0.1085 | True |
| MedQA test1273 | uniform_random | expected | 0.2500 expected |  |  |
| MedQA test1273 | label_prior_random | expected | 0.2530 expected |  |  |

## Paired Significance Table

| Comparison | Before Acc. | After Acc. | Delta Acc. | McNemar p | Before Collapse | After Collapse |
|---|---|---|---|---|---|---|
| 0.5B PubMedQA seed42 | 0.2100 | 0.6400 | +0.4300 | 6.03e-08 | True | False |
| 0.5B PubMedQA seed43 | 0.2100 | 0.6100 | +0.4000 | 8.96e-08 | True | False |
| 0.5B PubMedQA seed44 | 0.2100 | 0.7000 | +0.4900 | 3.16e-10 | True | False |
| 0.5B MedMCQA seed42 | 0.3170 | 0.3810 | +0.0640 | 0.0010 | True | False |
| 0.5B MedMCQA seed43 | 0.3170 | 0.4320 | +0.1150 | 9.30e-10 | True | False |
| 0.5B MedMCQA seed44 | 0.3170 | 0.4290 | +0.1120 | 8.91e-09 | True | False |
| 0.5B MedQA seed42 | 0.2875 | 0.3637 | +0.0762 | 5.66e-05 | True | False |
| 0.5B MedQA seed43 | 0.2875 | 0.3943 | +0.1068 | 6.80e-09 | True | False |
| 0.5B MedQA seed44 | 0.2875 | 0.4085 | +0.1210 | 3.39e-10 | True | False |
| 1.5B PubMedQA seed42 | 0.6200 | 0.6800 | +0.0600 | 0.3616 | True | False |
| 1.5B PubMedQA seed43 | 0.6200 | 0.7000 | +0.0800 | 0.2430 | True | False |
| 1.5B PubMedQA seed44 | 0.6200 | 0.6900 | +0.0700 | 0.2962 | True | False |
| 1.5B MedMCQA seed42 | 0.4360 | 0.5380 | +0.1020 | 1.11e-10 | False | False |
| 1.5B MedMCQA seed43 | 0.4360 | 0.5510 | +0.1150 | 7.24e-13 | False | False |
| 1.5B MedMCQA seed44 | 0.4360 | 0.5280 | +0.0920 | 5.88e-09 | False | False |
| 1.5B MedQA seed42 | 0.4698 | 0.5279 | +0.0581 | 9.90e-05 | False | False |
| 1.5B MedQA seed43 | 0.4698 | 0.5467 | +0.0770 | 9.06e-08 | False | False |
| 1.5B MedQA seed44 | 0.4698 | 0.5302 | +0.0605 | 3.60e-05 | False | False |
| 3B PubMedQA seed42 | 0.2300 | 0.6100 | +0.3800 | 3.24e-08 | True | False |
| 3B PubMedQA seed43 | 0.2300 | 0.6700 | +0.4400 | 3.71e-11 | True | False |
| 3B PubMedQA seed44 | 0.2300 | 0.6500 | +0.4200 | 1.28e-09 | True | False |
| 3B MedMCQA seed42 | 0.4900 | 0.5830 | +0.0930 | 1.20e-09 | False | False |
| 3B MedMCQA seed43 | 0.4900 | 0.5810 | +0.0910 | 1.18e-09 | False | False |
| 3B MedMCQA seed44 | 0.4900 | 0.5860 | +0.0960 | 4.91e-11 | False | False |
| 3B MedQA seed42 | 0.4965 | 0.5648 | +0.0683 | 3.70e-08 | False | False |
| 3B MedQA seed43 | 0.4965 | 0.5640 | +0.0676 | 2.27e-07 | False | False |
| 3B MedQA seed44 | 0.4965 | 0.5679 | +0.0715 | 6.85e-08 | False | False |
