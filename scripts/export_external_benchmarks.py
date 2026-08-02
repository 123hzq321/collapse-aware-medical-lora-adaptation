from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError


CHOICE_LABELS = ["A", "B", "C", "D"]
MMLU_MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_summary(path: Path, rows: list[dict], extra: dict | None = None) -> None:
    summary = {
        "rows": len(rows),
        "task_type_counts": dict(Counter(row.get("task_type", "") for row in rows)),
        "label_counts": dict(Counter(row.get("gold", "") for row in rows)),
    }
    if extra:
        summary.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def stable_id(prefix: str, *parts: object) -> str:
    payload = "\n".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def require_load_dataset():
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on local binary wheels
        raise RuntimeError(
            "Hugging Face datasets could not be imported. PubMedQA 5-fold export can run "
            "from local JSONL files, but MMLU/BoolQ/RTE export requires a working "
            "`datasets` + `pyarrow` installation."
        ) from exc
    return load_dataset


def fetch_hf_rows_api(
    dataset_name: str,
    config: str,
    split: str,
    max_rows: int | None = None,
    page_size: int = 100,
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        length = page_size
        if max_rows is not None:
            remaining = max_rows - len(rows)
            if remaining <= 0:
                break
            length = min(length, remaining)
        query = urlencode(
            {
                "dataset": dataset_name,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        for attempt in range(5):
            try:
                with urlopen(url, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                if exc.code != 429 or attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        page_rows = [item["row"] for item in payload.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
        time.sleep(0.2)
        if len(page_rows) < length:
            break
    return rows


def sample_rows(rows, max_samples: int | None, seed: int):
    if max_samples is None or max_samples >= len(rows):
        return rows
    if hasattr(rows, "shuffle") and hasattr(rows, "select"):
        return rows.shuffle(seed=seed).select(range(max_samples))
    copied = list(rows)
    random.Random(seed).shuffle(copied)
    return copied[:max_samples]


def local_pubmedqa_rows(root: Path) -> list[dict]:
    split_dir = root / "data" / "pubmedqa" / "splits"
    rows: list[dict] = []
    for split_name in ["train", "dev", "test"]:
        path = split_dir / f"{split_name}.jsonl"
        if not path.exists():
            return []
        for row in read_jsonl(path):
            rows.append(
                {
                    "id": f"pubmedqa-{row.get('id') or stable_id('row', row.get('question', ''))}",
                    "source_dataset": "qiaojin/PubMedQA",
                    "source_config": "pqa_labeled",
                    "source_split": split_name,
                    "benchmark": "pubmedqa_labeled_5fold",
                    "task_type": "yes_no_maybe",
                    "question": row["question"],
                    "context": row["context"],
                    "gold": row["gold"],
                    "labels": ["yes", "no", "maybe"],
                    "long_answer": row.get("long_answer", ""),
                }
            )
    return rows


def hf_pubmedqa_rows(seed: int) -> list[dict]:
    from medself.data import flatten_pubmedqa_context, load_pubmedqa

    datasets = load_pubmedqa(seed=seed)
    rows: list[dict] = []
    for split_name, split in datasets.items():
        for index, example in enumerate(split):
            rows.append(
                {
                    "id": f"pubmedqa-{example.get('pubid') or split_name + '-' + str(index)}",
                    "source_dataset": "qiaojin/PubMedQA",
                    "source_config": "pqa_labeled",
                    "source_split": split_name,
                    "benchmark": "pubmedqa_labeled_5fold",
                    "task_type": "yes_no_maybe",
                    "question": example["question"],
                    "context": flatten_pubmedqa_context(example),
                    "gold": example["final_decision"],
                    "labels": ["yes", "no", "maybe"],
                    "long_answer": example.get("long_answer", ""),
                }
            )
    return rows


def assign_stratified_folds(rows: list[dict], folds: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["gold"], []).append(row)

    assigned: list[dict] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        for index, row in enumerate(label_rows):
            out = dict(row)
            out["fold"] = index % folds
            assigned.append(out)
    assigned.sort(key=lambda row: (row["fold"], row["id"]))
    return assigned


def export_pubmedqa_5fold(root: Path, out_dir: Path, seed: int, folds: int) -> list[Path]:
    rows = local_pubmedqa_rows(root) or hf_pubmedqa_rows(seed=seed)
    assigned = assign_stratified_folds(rows, folds=folds, seed=seed)
    written: list[Path] = []

    all_path = out_dir / f"pubmedqa_labeled_all_seed{seed}.jsonl"
    write_jsonl(assigned, all_path)
    write_summary(
        out_dir / f"pubmedqa_labeled_all_seed{seed}_summary.json",
        assigned,
        {"dataset": "qiaojin/PubMedQA", "folds": folds, "seed": seed},
    )
    written.append(all_path)

    for fold in range(folds):
        test_rows = [row for row in assigned if row["fold"] == fold]
        train_rows = [row for row in assigned if row["fold"] != fold]
        test_path = out_dir / f"pubmedqa_labeled_fold{fold}_test_seed{seed}.jsonl"
        train_path = out_dir / f"pubmedqa_labeled_fold{fold}_train_seed{seed}.jsonl"
        write_jsonl(test_rows, test_path)
        write_jsonl(train_rows, train_path)
        write_summary(
            out_dir / f"pubmedqa_labeled_fold{fold}_summary.json",
            test_rows,
            {
                "dataset": "qiaojin/PubMedQA",
                "fold": fold,
                "train_rows": len(train_rows),
                "test_rows": len(test_rows),
                "seed": seed,
            },
        )
        written.extend([test_path, train_path])
    return written


def load_hf_split(dataset_name: str, config: str | None, split: str):
    config_for_api = config or "default"
    try:
        load_dataset = require_load_dataset()
        if config:
            return load_dataset(dataset_name, config, split=split)
        return load_dataset(dataset_name, split=split)
    except RuntimeError:
        return fetch_hf_rows_api(dataset_name, config_for_api, split)


def normalize_mmlu_choices(example: dict) -> dict[str, str] | None:
    choices = example.get("choices")
    if isinstance(choices, list) and len(choices) >= 4:
        return {label: str(choices[index]) for index, label in enumerate(CHOICE_LABELS)}
    if isinstance(choices, dict):
        out = {label: str(choices[label]) for label in CHOICE_LABELS if label in choices}
        return out if len(out) == 4 else None
    out = {}
    for label in CHOICE_LABELS:
        if label in example:
            out[label] = str(example[label])
    return out if len(out) == 4 else None


def normalize_answer_label(answer: object) -> str | None:
    if isinstance(answer, int) and 0 <= answer < len(CHOICE_LABELS):
        return CHOICE_LABELS[answer]
    text = str(answer).strip().upper()
    if text in CHOICE_LABELS:
        return text
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(CHOICE_LABELS):
            return CHOICE_LABELS[index]
    return None


def export_mmlu_medical(out_dir: Path, split: str, max_samples_per_subject: int | None, seed: int) -> list[Path]:
    rows: list[dict] = []
    for subject in MMLU_MEDICAL_SUBJECTS:
        dataset = load_hf_split("cais/mmlu", subject, split)
        dataset = sample_rows(dataset, max_samples_per_subject, seed)
        for index, example in enumerate(dataset):
            choices = normalize_mmlu_choices(example)
            gold = normalize_answer_label(example.get("answer"))
            if choices is None or gold is None:
                continue
            rows.append(
                {
                    "id": f"mmlu-{subject}-{split}-{index}",
                    "source_dataset": "cais/mmlu",
                    "source_config": subject,
                    "source_split": split,
                    "benchmark": "mmlu_medical",
                    "task_type": "mcq4",
                    "subject": subject,
                    "question": example["question"],
                    "choices": choices,
                    "gold": gold,
                    "labels": CHOICE_LABELS,
                }
            )

    path = out_dir / f"mmlu_medical_{split}_seed{seed}.jsonl"
    write_jsonl(rows, path)
    write_summary(
        out_dir / f"mmlu_medical_{split}_seed{seed}_summary.json",
        rows,
        {
            "dataset": "cais/mmlu",
            "subjects": MMLU_MEDICAL_SUBJECTS,
            "split": split,
            "seed": seed,
        },
    )
    return [path]


def export_boolq(out_dir: Path, split: str, max_samples: int | None, seed: int) -> list[Path]:
    dataset = load_hf_split("google/boolq", "default", split)
    dataset = sample_rows(dataset, max_samples, seed)
    rows = []
    for index, example in enumerate(dataset):
        rows.append(
            {
                "id": f"boolq-{split}-{index}",
                "source_dataset": "google/boolq",
                "source_split": split,
                "benchmark": "boolq_validation",
                "task_type": "yes_no",
                "question": example["question"],
                "context": example["passage"],
                "gold": "yes" if bool(example["answer"]) else "no",
                "labels": ["yes", "no"],
            }
        )
    path = out_dir / f"boolq_{split}_seed{seed}.jsonl"
    write_jsonl(rows, path)
    write_summary(
        out_dir / f"boolq_{split}_seed{seed}_summary.json",
        rows,
        {"dataset": "google/boolq", "split": split, "seed": seed},
    )
    return [path]


def export_rte(out_dir: Path, split: str, max_samples: int | None, seed: int) -> list[Path]:
    try:
        dataset = load_hf_split("nyu-mll/glue", "rte", split)
        source_dataset = "nyu-mll/glue"
    except Exception:
        dataset = load_hf_split("glue", "rte", split)
        source_dataset = "glue"

    dataset = sample_rows(dataset, max_samples, seed)

    rows = []
    for index, example in enumerate(dataset):
        label = int(example["label"])
        if label < 0:
            continue
        rows.append(
            {
                "id": f"rte-{split}-{index}",
                "source_dataset": source_dataset,
                "source_config": "rte",
                "source_split": split,
                "benchmark": "glue_rte_validation",
                "task_type": "nli2",
                "premise": example["sentence1"],
                "hypothesis": example["sentence2"],
                "gold": "entailment" if label == 0 else "not_entailment",
                "labels": ["entailment", "not_entailment"],
            }
        )
    path = out_dir / f"glue_rte_{split}_seed{seed}.jsonl"
    write_jsonl(rows, path)
    write_summary(
        out_dir / f"glue_rte_{split}_seed{seed}_summary.json",
        rows,
        {"dataset": source_dataset, "config": "rte", "split": split, "seed": seed},
    )
    return [path]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="data/external")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["pubmedqa_5fold", "mmlu_medical", "boolq", "rte"],
        choices=["pubmedqa_5fold", "mmlu_medical", "boolq", "rte"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mmlu-split", default="test")
    parser.add_argument("--boolq-split", default="validation")
    parser.add_argument("--rte-split", default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-subject", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / args.out_dir
    written: list[Path] = []

    if "pubmedqa_5fold" in args.tasks:
        written.extend(export_pubmedqa_5fold(root, out_dir, seed=args.seed, folds=args.folds))
    if "mmlu_medical" in args.tasks:
        written.extend(
            export_mmlu_medical(
                out_dir,
                split=args.mmlu_split,
                max_samples_per_subject=args.max_samples_per_subject,
                seed=args.seed,
            )
        )
    if "boolq" in args.tasks:
        written.extend(export_boolq(out_dir, split=args.boolq_split, max_samples=args.max_samples, seed=args.seed))
    if "rte" in args.tasks:
        written.extend(export_rte(out_dir, split=args.rte_split, max_samples=args.max_samples, seed=args.seed))

    print(json.dumps({"written": [str(path) for path in written]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
