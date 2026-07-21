# Data Split And Leakage Audit

Exact ID and normalized exact-question overlap are computed between training sources and evaluation files. This audit does not prove semantic independence; it checks file-level split separation and exact duplicate contamination.

| comparison | train_n | eval_n | train_duplicate_question_n | exact_id_overlap_n | exact_question_overlap_n | note |
|---|---|---|---|---|---|---|
| pubmedqa_train_vs_test | 800 | 100 | 0 | 0 | 0 | Original PubMedQA train/test files. |
| medmcqa_train30k_vs_val1000 | 30000 | 1000 | 1 | 0 | 0 | MedMCQA train subset versus validation subset. |
| medqa_train_vs_test | 10178 | 1273 | 2 | 0 | 0 | MedQA train/test files. |
| full45k_vs_pubmedqa_test | 4800 | 100 | 4000 | 0 | 0 | Full45k mixture versus PubMedQA test. Train duplicates include intentional PubMedQA replay. |
| full45k_vs_medmcqa_val1000 | 30000 | 1000 | 1 | 0 | 0 | Full45k mixture versus MedMCQA validation. |
| full45k_vs_medqa_test | 10178 | 1273 | 2 | 0 | 0 | Full45k mixture versus MedQA test. |
