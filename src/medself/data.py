from __future__ import annotations

from datasets import DatasetDict, load_dataset


def load_pubmedqa(seed: int = 42) -> DatasetDict:
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    dev_test = split["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(
        {
            "train": split["train"],
            "dev": dev_test["train"],
            "test": dev_test["test"],
        }
    )


def flatten_pubmedqa_context(example: dict) -> str:
    context = example["context"]
    if isinstance(context, dict) and "contexts" in context:
        return "\n".join(context["contexts"])
    if isinstance(context, list):
        return "\n".join(str(item) for item in context)
    return str(context)
