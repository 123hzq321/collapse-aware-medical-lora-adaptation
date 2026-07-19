| task_name | family | model | condition | n_runs | seeds | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std | dominant_rate_mean | entropy_mean | collapse_count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MedMCQA val1000 | frozen_baseline | qwen05b | base | 1 |  | 0.3170 | 0.0000 | 0.1672 | 0.0000 | 0.8840 | 0.2703 | 1 |
| MedMCQA val1000 | frozen_baseline | qwen15b | base | 1 |  | 0.4360 | 0.0000 | 0.4252 | 0.0000 | 0.3060 | 0.9838 | 0 |
| MedMCQA val1000 | frozen_baseline | qwen3b | base | 1 |  | 0.4900 | 0.0000 | 0.4851 | 0.0000 | 0.2910 | 0.9837 | 0 |
| MedMCQA val1000 | self_update | qwen05b | full45k | 3 | 42 43 44 | 0.4140 | 0.0286 | 0.3973 | 0.0297 | 0.3920 | 0.9447 | 0 |
| MedMCQA val1000 | self_update | qwen15b | full45k | 3 | 42 43 44 | 0.5390 | 0.0115 | 0.5282 | 0.0112 | 0.3697 | 0.9679 | 0 |
| MedMCQA val1000 | self_update | qwen3b | full45k | 3 | 42 43 44 | 0.5833 | 0.0025 | 0.5762 | 0.0013 | 0.3510 | 0.9723 | 0 |
| MedQA test1273 | frozen_baseline | qwen05b | base | 1 |  | 0.2875 | 0.0000 | 0.1712 | 0.0000 | 0.8335 | 0.3556 | 1 |
| MedQA test1273 | frozen_baseline | qwen15b | base | 1 |  | 0.4698 | 0.0000 | 0.4673 | 0.0000 | 0.3111 | 0.9924 | 0 |
| MedQA test1273 | frozen_baseline | qwen3b | base | 1 |  | 0.4965 | 0.0000 | 0.4956 | 0.0000 | 0.2742 | 0.9981 | 0 |
| MedQA test1273 | self_update | qwen05b | full45k | 3 | 42 43 44 | 0.3888 | 0.0229 | 0.3805 | 0.0244 | 0.3231 | 0.9803 | 0 |
| MedQA test1273 | self_update | qwen15b | full45k | 3 | 42 43 44 | 0.5350 | 0.0103 | 0.5311 | 0.0102 | 0.2783 | 0.9935 | 0 |
| MedQA test1273 | self_update | qwen3b | full45k | 3 | 42 43 44 | 0.5656 | 0.0021 | 0.5623 | 0.0024 | 0.2799 | 0.9951 | 0 |
| PubMedQA test100 | frozen_baseline | qwen05b | base | 1 |  | 0.2100 | 0.0000 | 0.1741 | 0.0000 | 0.8300 | 0.4150 | 1 |
| PubMedQA test100 | frozen_baseline | qwen15b | base | 1 |  | 0.6200 | 0.0000 | 0.4058 | 0.0000 | 0.8500 | 0.4717 | 1 |
| PubMedQA test100 | frozen_baseline | qwen3b | base | 1 |  | 0.2300 | 0.0000 | 0.2454 | 0.0000 | 0.8100 | 0.5564 | 1 |
| PubMedQA test100 | self_update | qwen05b | full45k | 3 | 42 43 44 | 0.6500 | 0.0458 | 0.4978 | 0.0626 | 0.6367 | 0.7661 | 0 |
| PubMedQA test100 | self_update | qwen15b | full45k | 3 | 42 43 44 | 0.6900 | 0.0100 | 0.5842 | 0.0402 | 0.5433 | 0.8973 | 0 |
| PubMedQA test100 | self_update | qwen3b | full45k | 3 | 42 43 44 | 0.6433 | 0.0306 | 0.5709 | 0.0450 | 0.4833 | 0.9475 | 0 |
