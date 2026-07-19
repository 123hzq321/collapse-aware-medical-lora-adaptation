# Full-Matrix 结果记录

日期：2026-07-14

## 当前状态

已完成：

- qwen05b full45k seed42：训练完成，三项评估完成。
- qwen05b full45k seed43：训练完成，三项评估完成。
- qwen05b full45k seed44：训练完成，三项评估完成。
- qwen05b baseline：PubMedQA test100、MedMCQA val1000、MedQA test1273 完成。
- qwen15b full45k seed42：训练完成，三项评估完成。
- qwen15b full45k seed43：训练完成，三项评估完成。
- qwen15b full45k seed44：训练完成，三项评估完成。
- qwen15b baseline：PubMedQA test100、MedMCQA val1000、MedQA test1273 完成。
- qwen3b baseline：PubMedQA test100、MedMCQA val1000、MedQA test1273 完成。
- qwen3b full45k seed42：训练完成，三项评估完成。
- qwen3b full45k seed43：训练完成，三项评估完成。
- qwen3b full45k seed44：训练完成，三项评估完成。
- qwen15b smoke train：8 samples 成功。
- qwen3b smoke train：4 samples 成功。

正在运行：

- 暂无。

待运行：

- 暂无。

## qwen05b seed42 full45k

训练：

| 项 | 值 |
|---|---:|
| train rows | 44,978 |
| optimizer steps | 5,623 |
| max length | 1536 |
| LoRA r | 8 |
| trainable params | 4,399,104 |
| training time | 1:39:27 |

## qwen05b baseline vs full45k seed42

| 任务 | 模型 | accuracy | macro-F1 | dominant label | dominant rate | entropy | collapse |
|---|---|---:|---:|---|---:|---:|---|
| PubMedQA test100 | baseline | 0.210 | 0.174 | maybe | 0.830 | 0.415 | yes |
| PubMedQA test100 | full45k seed42 | 0.640 | 0.442 | yes | 0.660 | 0.713 | no |
| MedMCQA val1000 | baseline | 0.317 | 0.167 | A | 0.884 | 0.270 | yes |
| MedMCQA val1000 | full45k seed42 | 0.381 | 0.363 | A | 0.372 | 0.941 | no |
| MedQA test1273 | baseline | 0.288 | 0.171 | A | 0.833 | 0.356 | yes |
| MedQA test1273 | full45k seed42 | 0.364 | 0.354 | B | 0.351 | 0.977 | no |

## qwen05b full45k 三 seed 聚合

| 任务 | condition | n | accuracy | macro-F1 | dominant rate | entropy | collapse |
|---|---|---:|---:|---:|---:|---:|---:|
| PubMedQA test100 | baseline | 1 | 0.210 | 0.174 | 0.830 | 0.415 | 1/1 |
| PubMedQA test100 | full45k | 3 | 0.650 ± 0.046 | 0.498 ± 0.063 | 0.637 ± 0.021 | 0.766 ± 0.059 | 0/3 |
| MedMCQA val1000 | baseline | 1 | 0.317 | 0.167 | 0.884 | 0.270 | 1/1 |
| MedMCQA val1000 | full45k | 3 | 0.414 ± 0.029 | 0.397 ± 0.030 | 0.392 ± 0.035 | 0.945 ± 0.011 | 0/3 |
| MedQA test1273 | baseline | 1 | 0.288 | 0.171 | 0.833 | 0.356 | 1/1 |
| MedQA test1273 | full45k | 3 | 0.389 ± 0.023 | 0.381 ± 0.024 | 0.323 ± 0.030 | 0.980 ± 0.004 | 0/3 |

逐 seed 明细见：

```text
work/med-self-update/runs/collapse_qwen05b_full45k_3seed.md
work/med-self-update/runs/aggregate_qwen05b_full45k_3seed.md
```

## qwen15b baseline vs full45k 三 seed

训练：

| 项 | seed42 | seed43 | seed44 |
|---|---:|---:|---:|
| train rows | 44,978 | 44,978 | 44,978 |
| optimizer steps | 5,623 | 5,623 | 5,623 |
| max length | 1024 | 1024 | 1024 |
| LoRA r | 8 | 8 | 8 |
| trainable params | 9,232,384 | 9,232,384 | 9,232,384 |
| gradient checkpointing | yes | yes | yes |
| training time | 3:33:36 | 3:18:02 | 3:21:30 |

| 任务 | condition | n | accuracy | macro-F1 | dominant rate | entropy | collapse |
|---|---|---:|---:|---:|---:|---:|---:|
| PubMedQA test100 | baseline | 1 | 0.620 | 0.406 | 0.850 | 0.472 | 1/1 |
| PubMedQA test100 | full45k | 3 | 0.690 ± 0.010 | 0.584 ± 0.040 | 0.543 ± 0.040 | 0.897 ± 0.040 | 0/3 |
| MedMCQA val1000 | baseline | 1 | 0.436 | 0.425 | 0.306 | 0.984 | 0/1 |
| MedMCQA val1000 | full45k | 3 | 0.539 ± 0.012 | 0.528 ± 0.011 | 0.370 ± 0.008 | 0.968 ± 0.003 | 0/3 |
| MedQA test1273 | baseline | 1 | 0.470 | 0.467 | 0.311 | 0.992 | 0/1 |
| MedQA test1273 | full45k | 3 | 0.535 ± 0.010 | 0.531 ± 0.010 | 0.278 ± 0.005 | 0.993 ± 0.002 | 0/3 |

明细见：

```text
work/med-self-update/runs/collapse_qwen15b_full45k_3seed.md
work/med-self-update/runs/aggregate_qwen15b_full45k_3seed.md
```

## qwen3b baseline vs full45k 三 seed

训练：

| 项 | seed42 | seed43 | seed44 |
|---|---:|---:|---:|
| train rows | 44,978 | 44,978 | 44,978 |
| optimizer steps | 2,812 | 2,812 | 2,812 |
| max length | 768 | 768 | 768 |
| LoRA r | 8 | 8 | 8 |
| trainable params | 14,966,784 | 14,966,784 | 14,966,784 |
| gradient checkpointing | yes | yes | yes |
| training time | 4:51:37 | 4:22:25 | 5:53:31 |

| 任务 | condition | n | accuracy | macro-F1 | dominant rate | entropy | collapse |
|---|---|---:|---:|---:|---:|---:|---:|
| PubMedQA test100 | baseline | 1 | 0.230 | 0.245 | 0.810 | 0.556 | 1/1 |
| PubMedQA test100 | full45k | 3 | 0.643 ± 0.031 | 0.571 ± 0.045 | 0.483 ± 0.025 | 0.947 ± 0.021 | 0/3 |
| MedMCQA val1000 | baseline | 1 | 0.490 | 0.485 | 0.291 | 0.984 | 0/1 |
| MedMCQA val1000 | full45k | 3 | 0.583 ± 0.003 | 0.576 ± 0.001 | 0.351 ± 0.007 | 0.972 ± 0.004 | 0/3 |
| MedQA test1273 | baseline | 1 | 0.496 | 0.496 | 0.274 | 0.998 | 0/1 |
| MedQA test1273 | full45k | 3 | 0.566 ± 0.002 | 0.562 ± 0.002 | 0.280 ± 0.005 | 0.995 ± 0.002 | 0/3 |

明细见：

```text
work/med-self-update/runs/collapse_qwen3b_base.md
work/med-self-update/runs/collapse_qwen3b_full45k_3seed.md
work/med-self-update/runs/aggregate_qwen3b_full45k_3seed.md
```

## 当前 scale matrix 聚合

| 任务 | condition | n | accuracy | macro-F1 | dominant rate | entropy | collapse |
|---|---|---:|---:|---:|---:|---:|---:|
| PubMedQA test100 | qwen05b_base | 1 | 0.210 | 0.174 | 0.830 | 0.415 | 1/1 |
| PubMedQA test100 | qwen05b_full45k | 3 | 0.650 ± 0.046 | 0.498 ± 0.063 | 0.637 ± 0.021 | 0.766 ± 0.059 | 0/3 |
| PubMedQA test100 | qwen15b_base | 1 | 0.620 | 0.406 | 0.850 | 0.472 | 1/1 |
| PubMedQA test100 | qwen15b_full45k | 3 | 0.690 ± 0.010 | 0.584 ± 0.040 | 0.543 ± 0.040 | 0.897 ± 0.040 | 0/3 |
| PubMedQA test100 | qwen3b_base | 1 | 0.230 | 0.245 | 0.810 | 0.556 | 1/1 |
| PubMedQA test100 | qwen3b_full45k | 3 | 0.643 ± 0.031 | 0.571 ± 0.045 | 0.483 ± 0.025 | 0.947 ± 0.021 | 0/3 |
| MedMCQA val1000 | qwen05b_base | 1 | 0.317 | 0.167 | 0.884 | 0.270 | 1/1 |
| MedMCQA val1000 | qwen05b_full45k | 3 | 0.414 ± 0.029 | 0.397 ± 0.030 | 0.392 ± 0.035 | 0.945 ± 0.011 | 0/3 |
| MedMCQA val1000 | qwen15b_base | 1 | 0.436 | 0.425 | 0.306 | 0.984 | 0/1 |
| MedMCQA val1000 | qwen15b_full45k | 3 | 0.539 ± 0.012 | 0.528 ± 0.011 | 0.370 ± 0.008 | 0.968 ± 0.003 | 0/3 |
| MedMCQA val1000 | qwen3b_base | 1 | 0.490 | 0.485 | 0.291 | 0.984 | 0/1 |
| MedMCQA val1000 | qwen3b_full45k | 3 | 0.583 ± 0.003 | 0.576 ± 0.001 | 0.351 ± 0.007 | 0.972 ± 0.004 | 0/3 |
| MedQA test1273 | qwen05b_base | 1 | 0.288 | 0.171 | 0.833 | 0.356 | 1/1 |
| MedQA test1273 | qwen05b_full45k | 3 | 0.389 ± 0.023 | 0.381 ± 0.024 | 0.323 ± 0.030 | 0.980 ± 0.004 | 0/3 |
| MedQA test1273 | qwen15b_base | 1 | 0.470 | 0.467 | 0.311 | 0.992 | 0/1 |
| MedQA test1273 | qwen15b_full45k | 3 | 0.535 ± 0.010 | 0.531 ± 0.010 | 0.278 ± 0.005 | 0.993 ± 0.002 | 0/3 |
| MedQA test1273 | qwen3b_base | 1 | 0.496 | 0.496 | 0.274 | 0.998 | 0/1 |
| MedQA test1273 | qwen3b_full45k | 3 | 0.566 ± 0.002 | 0.562 ± 0.002 | 0.280 ± 0.005 | 0.995 ± 0.002 | 0/3 |

机器可读汇总见：

```text
work/med-self-update/runs/collapse_scale_matrix_current.md
work/med-self-update/runs/aggregate_scale_matrix_current.md
```

## Baseline 与 BENCHMARKER

已补充非模型 baseline 和统一 BENCHMARKER：

```text
outputs/baseline_and_benchmarker.md
work/med-self-update/scripts/benchmarker.py
work/med-self-update/runs/benchmarker_current.md
work/med-self-update/runs/benchmarker_baselines_current.md
work/med-self-update/runs/benchmarker_aggregate_current.md
work/med-self-update/runs/benchmarker_predictions_current.csv
```

非模型 baseline 摘要：

| 任务 | majority baseline | accuracy | macro-F1 | collapse |
|---|---|---:|---:|---|
| PubMedQA test100 | always yes | 0.560 | 0.239 | yes |
| MedMCQA val1000 | always A | 0.323 | 0.122 | yes |
| MedQA test1273 | always A | 0.277 | 0.109 | yes |

BENCHMARKER 当前最佳 self-update：

| 任务 | 最佳模型 | accuracy | macro-F1 | collapse |
|---|---|---:|---:|---|
| PubMedQA test100 | qwen15b full45k 3-seed mean | 0.690 | 0.584 | 0/3 |
| MedMCQA val1000 | qwen3b full45k 3-seed mean | 0.583 | 0.576 | 0/3 |
| MedQA test1273 | qwen3b full45k 3-seed mean | 0.566 | 0.562 | 0/3 |

## Paired 统计分析

McNemar 检验基于同一测试集上的逐样本正确/错误变化；因此比只看两个 accuracy 更适合回答“训练后是否真的修正了样本”。完整表见：

```text
work/med-self-update/runs/paired_stats_current.md
work/med-self-update/runs/paired_stats_current.csv
```

关键结论：

1. qwen05b full45k 相对 qwen05b baseline 在三任务三 seed 上均显著提升 accuracy，并且全部解除 collapse。PubMedQA 三个 seed 的 delta accuracy 为 +0.400 到 +0.490，McNemar p 均小于 1e-7；MedMCQA 为 +0.064 到 +0.115，MedQA 为 +0.076 到 +0.121。
2. qwen15b full45k 在 MedMCQA 和 MedQA 上三个 seed 均显著提升：MedMCQA delta accuracy 为 +0.092 到 +0.115，McNemar p 均小于 6e-9；MedQA 为 +0.058 到 +0.077，McNemar p 均小于 1e-4。
3. qwen15b PubMedQA 三个 seed 的 accuracy 从 0.620 提升到 0.680-0.700，但 McNemar p=0.243-0.362，不应夸大为显著 accuracy 提升；更稳妥的说法是 mean entropy 从 0.472 到 0.897，mean dominant rate 从 0.850 到 0.543，collapse 从 1/1 降到 0/3。
4. qwen3b full45k 相对 qwen3b baseline 在三任务三 seed 上均显著提升：PubMedQA delta accuracy 为 +0.380 到 +0.440，MedMCQA 为 +0.091 到 +0.096，MedQA 为 +0.068 到 +0.071；PubMedQA collapse 从 1/1 降到 0/3。
5. 跨尺度结论现在更清楚：3B 在 MedMCQA/MedQA 上最强且三 seed 方差很小；PubMedQA 最强仍是 1.5B，说明 yes/no/maybe 任务受标签分布和任务格式影响，不是单纯参数越多越好。

## 初步观察

1. full45k 在 qwen05b、qwen15b、qwen3b 的三个 seed 上都解除 collapse：三项任务的 full45k collapse_count 均为 0/3。
2. MedMCQA 和 MedQA 的 macro-F1 大幅提高，说明不是只换了 dominant label，而是 A/B/C/D 覆盖更均衡。
3. 1.5B full45k 三 seed 在三项任务上均超过 1.5B baseline，并且把 PubMedQA 的 yes-dominant collapse 从 0.850 降到 mean dominant rate 0.543。
4. 3B baseline 在 MedMCQA/MedQA 上强于 1.5B baseline，但 PubMedQA 反而出现 maybe-dominant collapse；full45k 后 3B PubMedQA dominant rate 从 0.810 降到 0.483 ± 0.025，说明自更新数据能解除这种格式性坍缩。
5. PubMedQA 上 0.5B full45k 三 seed accuracy 为 0.650 ± 0.046，1.5B 为 0.690 ± 0.010，3B 为 0.643 ± 0.031；这提示任务配比和模型尺度共同影响 yes/no/maybe 专门能力。
6. PubMedQA 的 maybe 类仍然脆弱，后续需要任务配比、class-balanced replay 或不确定性 replay 来专门处理。

## 论文价值

当前最强证据不是“full45k 一定比 8k 更好”，而是：

> 数据扩容和跨任务 replay 能显著解除小模型在医学多任务 SFT 中的单标签坍缩；但尺度增大并不能天然解决 PubMedQA 的 yes/no/maybe 偏置，任务混合比例仍会带来跨任务干扰，尤其影响低频不确定标签。

这比单纯刷 accuracy 更适合 BIBM/BMC/JMIR 的叙事。
