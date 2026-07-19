# Baseline 与 BENCHMARKER 补充

日期：2026-07-14

## 已补内容

1. 新增 `scripts/benchmarker.py`，作为统一 BENCHMARKER 入口。
2. 补充非模型 baseline：gold-majority、uniform-random expected、label-prior-random expected。
3. 将 frozen model baseline、full45k self-update、collapse、leaderboard 汇总到一套 benchmarker artifact。

## 非模型 Baseline

| 任务 | baseline | accuracy | macro-F1 / expected acc | collapse |
|---|---|---:|---:|---|
| PubMedQA test100 | majority always yes | 0.560 | macro-F1 0.239 | yes |
| PubMedQA test100 | uniform random | 0.333 expected | - | - |
| PubMedQA test100 | label-prior random | 0.439 expected | - | - |
| MedMCQA val1000 | majority always A | 0.323 | macro-F1 0.122 | yes |
| MedMCQA val1000 | uniform random | 0.250 expected | - | - |
| MedMCQA val1000 | label-prior random | 0.260 expected | - | - |
| MedQA test1273 | majority always A | 0.277 | macro-F1 0.109 | yes |
| MedQA test1273 | uniform random | 0.250 expected | - | - |
| MedQA test1273 | label-prior random | 0.253 expected | - | - |

## BENCHMARKER 结论

当前最佳 self-update 结果：

| 任务 | 最佳模型 | accuracy | macro-F1 | collapse |
|---|---|---:|---:|---|
| PubMedQA test100 | qwen15b full45k 3-seed mean | 0.690 | 0.584 | 0/3 |
| MedMCQA val1000 | qwen3b full45k 3-seed mean | 0.583 | 0.576 | 0/3 |
| MedQA test1273 | qwen3b full45k 3-seed mean | 0.566 | 0.562 | 0/3 |

关键解释：

1. majority baseline 自身都是 collapse，因此只看 accuracy 不够，必须同时报告 macro-F1、dominant-rate 和 entropy。
2. qwen05b frozen baseline 在 MedMCQA/MedQA 上接近或低于 majority baseline，并且 dominant-rate 极高，说明小模型冻结推理有明显单标签坍缩。
3. full45k 后，0.5B、1.5B、3B 的三个 seed 均解除 collapse；这比单纯 accuracy 提升更能支撑论文主张。
4. qwen3b baseline 在 PubMedQA 上仍 collapse 到 maybe，说明模型变大不是充分条件；full45k 后 dominant-rate 从 0.810 降到 0.483 ± 0.025，说明自更新数据对格式性坍缩有效。

## Artifact

```text
work/med-self-update/scripts/benchmarker.py
work/med-self-update/runs/benchmarker_current.md
work/med-self-update/runs/benchmarker_baselines_current.md
work/med-self-update/runs/benchmarker_aggregate_current.md
work/med-self-update/runs/benchmarker_predictions_current.csv
work/med-self-update/runs/benchmarker_all_current.csv
```

复现命令：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python scripts\benchmarker.py --root . --out-prefix runs\benchmarker_current
```
