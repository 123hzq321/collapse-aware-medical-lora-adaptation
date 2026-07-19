| task_name | family | model | condition | n | accuracy | macro_f1 | expected_accuracy | collapse_flag |
|---|---|---|---|---|---|---|---|---|
| PubMedQA test100 | label_baseline | gold_majority | always_yes | 100 | 0.5600 | 0.2393 |  | True |
| PubMedQA test100 | label_baseline | uniform_random | expected | 100 |  |  | 0.3333 |  |
| PubMedQA test100 | label_baseline | label_prior_random | expected | 100 |  |  | 0.4392 |  |
| MedMCQA val1000 | label_baseline | gold_majority | always_A | 1000 | 0.3230 | 0.1221 |  | True |
| MedMCQA val1000 | label_baseline | uniform_random | expected | 1000 |  |  | 0.2500 |  |
| MedMCQA val1000 | label_baseline | label_prior_random | expected | 1000 |  |  | 0.2601 |  |
| MedQA test1273 | label_baseline | gold_majority | always_A | 1273 | 0.2773 | 0.1085 |  | True |
| MedQA test1273 | label_baseline | uniform_random | expected | 1273 |  |  | 0.2500 |  |
| MedQA test1273 | label_baseline | label_prior_random | expected | 1273 |  |  | 0.2530 |  |
