from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from medself.prompts import build_medmcqa_prompt, build_pubmedqa_prompt


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_pubmedqa_rows(path: Path, tokenizer, repeat: int) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        prompt = build_pubmedqa_prompt(tokenizer, row["question"], row["context"])
        rows.append(
            {
                "source": "pubmedqa",
                "task": "yes_no_maybe",
                "id": row["id"],
                "prompt": prompt,
                "target": row["gold"],
                "gold": row["gold"],
            }
        )
    return rows * repeat


def make_mcq_rows(path: Path, tokenizer, max_rows: int | None, source: str) -> list[dict]:
    raw_rows = read_jsonl(path)
    if max_rows is not None:
        raw_rows = raw_rows[:max_rows]

    rows = []
    for row in raw_rows:
        prompt = build_medmcqa_prompt(tokenizer, row["question"], row["choices"])
        rows.append(
            {
                "source": source,
                "task": "multiple_choice",
                "id": row["id"],
                "prompt": prompt,
                "target": row["gold"],
                "gold": row["gold"],
                "subject_name": row.get("subject_name", ""),
                "topic_name": row.get("topic_name", ""),
                "meta_info": row.get("meta_info", ""),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--pubmedqa-train", default="data/pubmedqa/splits/train.jsonl")
    parser.add_argument("--medmcqa-train", default="data/medmcqa/train_10000_seed42.jsonl")
    parser.add_argument("--max-medmcqa", type=int, default=None)
    parser.add_argument("--medqa-train", default=None)
    parser.add_argument("--max-medqa", type=int, default=None)
    parser.add_argument("--pubmedqa-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/multitask/sft_pubmedqa_x3_medmcqa10k_seed42.jsonl")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = []
    rows.extend(make_pubmedqa_rows(Path(args.pubmedqa_train), tokenizer, args.pubmedqa_repeat))
    rows.extend(make_mcq_rows(Path(args.medmcqa_train), tokenizer, args.max_medmcqa, "medmcqa"))
    if args.medqa_train:
        rows.extend(make_mcq_rows(Path(args.medqa_train), tokenizer, args.max_medqa, "medqa"))

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "path": str(out_path),
        "rows": len(rows),
        "seed": args.seed,
        "pubmedqa_train": args.pubmedqa_train,
        "pubmedqa_repeat": args.pubmedqa_repeat,
        "medmcqa_train": args.medmcqa_train,
        "max_medmcqa": args.max_medmcqa,
        "medqa_train": args.medqa_train,
        "max_medqa": args.max_medqa,
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "task_counts": dict(Counter(row["task"] for row in rows)),
        "target_counts_by_source": {
            source: dict(Counter(row["target"] for row in rows if row["source"] == source))
            for source in sorted({row["source"] for row in rows})
        },
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
