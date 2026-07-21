from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


COMPARISONS = [
    {
        "name": "pubmedqa_train_vs_test",
        "task": "pubmedqa",
        "train": "data/pubmedqa/splits/train.jsonl",
        "eval": "data/pubmedqa/splits/test.jsonl",
        "train_source": "",
        "eval_source": "",
        "note": "Original PubMedQA train/test files.",
    },
    {
        "name": "medmcqa_train30k_vs_val1000",
        "task": "medmcqa",
        "train": "data/medmcqa/train_30000_seed42.jsonl",
        "eval": "data/medmcqa/validation_1000_seed42.jsonl",
        "train_source": "",
        "eval_source": "",
        "note": "MedMCQA train subset versus validation subset.",
    },
    {
        "name": "medqa_train_vs_test",
        "task": "medqa",
        "train": "data/medqa/train_full_seed42.jsonl",
        "eval": "data/medqa/test_full_seed42.jsonl",
        "train_source": "",
        "eval_source": "",
        "note": "MedQA train/test files.",
    },
    {
        "name": "full45k_vs_pubmedqa_test",
        "task": "pubmedqa",
        "train": "data/multitask/sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42.jsonl",
        "eval": "data/pubmedqa/splits/test.jsonl",
        "train_source": "pubmedqa",
        "eval_source": "pubmedqa",
        "note": "Full45k mixture versus PubMedQA test. Train duplicates include intentional PubMedQA replay.",
    },
    {
        "name": "full45k_vs_medmcqa_val1000",
        "task": "medmcqa",
        "train": "data/multitask/sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42.jsonl",
        "eval": "data/medmcqa/validation_1000_seed42.jsonl",
        "train_source": "medmcqa",
        "eval_source": "medmcqa",
        "note": "Full45k mixture versus MedMCQA validation.",
    },
    {
        "name": "full45k_vs_medqa_test",
        "task": "medqa",
        "train": "data/multitask/sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42.jsonl",
        "eval": "data/medqa/test_full_seed42.jsonl",
        "train_source": "medqa",
        "eval_source": "medqa",
        "note": "Full45k mixture versus MedQA test.",
    },
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_question(row: dict) -> str:
    if row.get("question"):
        return str(row["question"])
    prompt = str(row.get("prompt", ""))
    marker = "Question:\n"
    if marker not in prompt:
        return ""
    tail = prompt.split(marker, 1)[1]
    for stop in ("\n\nContext:", "\n\nOptions:", "\n\nReturn exactly", "\nAnswer:"):
        if stop in tail:
            return tail.split(stop, 1)[0]
    return tail.strip()


def id_key(row: dict, source: str = "") -> str:
    row_source = str(row.get("source") or source or "")
    row_id = str(row.get("id") or row.get("pubid") or row.get("index") or "")
    return f"{row_source}:{row_id}" if row_source else row_id


def filter_rows(rows: list[dict], source: str) -> list[dict]:
    if not source:
        return rows
    return [row for row in rows if str(row.get("source", source)) == source]


def duplicate_count(values: list[str]) -> int:
    non_empty = [value for value in values if value]
    return len(non_empty) - len(set(non_empty))


def audit_one(root: Path, comparison: dict) -> dict:
    train_path = root / comparison["train"]
    eval_path = root / comparison["eval"]
    train_rows = filter_rows(read_jsonl(train_path), comparison["train_source"])
    eval_rows = filter_rows(read_jsonl(eval_path), comparison["eval_source"])

    train_ids = [id_key(row, comparison["train_source"]) for row in train_rows]
    eval_ids = [id_key(row, comparison["eval_source"]) for row in eval_rows]
    train_questions = [normalize_text(extract_question(row)) for row in train_rows]
    eval_questions = [normalize_text(extract_question(row)) for row in eval_rows]

    id_overlap = sorted(set(train_ids) & set(eval_ids))
    question_overlap = sorted(set(train_questions) & set(eval_questions) - {""})

    return {
        "comparison": comparison["name"],
        "task": comparison["task"],
        "train_file": comparison["train"],
        "eval_file": comparison["eval"],
        "train_n": len(train_rows),
        "eval_n": len(eval_rows),
        "train_duplicate_id_n": duplicate_count(train_ids),
        "eval_duplicate_id_n": duplicate_count(eval_ids),
        "train_duplicate_question_n": duplicate_count(train_questions),
        "eval_duplicate_question_n": duplicate_count(eval_questions),
        "exact_id_overlap_n": len(id_overlap),
        "exact_question_overlap_n": len(question_overlap),
        "example_id_overlap": "; ".join(id_overlap[:3]),
        "example_question_overlap": " || ".join(question_overlap[:2]),
        "note": comparison["note"],
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = [
        "comparison",
        "train_n",
        "eval_n",
        "train_duplicate_question_n",
        "exact_id_overlap_n",
        "exact_question_overlap_n",
        "note",
    ]
    lines = [
        "# Data Split And Leakage Audit",
        "",
        "Exact ID and normalized exact-question overlap are computed between training sources and evaluation files. This audit does not prove semantic independence; it checks file-level split separation and exact duplicate contamination.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]).replace("|", "/") for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-prefix", default="runs/data_split_audit_current")
    args = parser.parse_args()

    root = Path(args.root)
    out_prefix = root / args.out_prefix
    rows = [audit_one(root, comparison) for comparison in COMPARISONS]
    write_csv(rows, out_prefix.with_suffix(".csv"))
    write_markdown(rows, out_prefix.with_suffix(".md"))
    print(f"wrote split audit to {out_prefix.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
