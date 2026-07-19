from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


CHOICE_KEYS = ["opa", "opb", "opc", "opd"]
CHOICE_LABELS = ["A", "B", "C", "D"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/medmcqa")
    args = parser.parse_args()

    dataset = load_dataset("openlifescienceai/medmcqa")[args.split]
    if args.max_samples < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_{len(dataset)}_seed{args.seed}.jsonl"

    counts = Counter()
    subjects = Counter()
    with out_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(dataset):
            answer_index = int(example["cop"])
            answer_label = CHOICE_LABELS[answer_index]
            counts[answer_label] += 1
            subjects[example.get("subject_name") or ""] += 1
            row = {
                "id": example["id"],
                "index": index,
                "source_split": args.split,
                "question": example["question"],
                "choices": {
                    label: example[key]
                    for label, key in zip(CHOICE_LABELS, CHOICE_KEYS)
                },
                "gold": answer_label,
                "gold_index": answer_index,
                "explanation": example.get("exp", ""),
                "choice_type": example.get("choice_type", ""),
                "subject_name": example.get("subject_name", ""),
                "topic_name": example.get("topic_name", ""),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "openlifescienceai/medmcqa",
        "split": args.split,
        "rows": len(dataset),
        "seed": args.seed,
        "path": str(out_path),
        "label_counts": dict(counts),
        "top_subjects": dict(subjects.most_common(20)),
    }
    summary_path = out_dir / f"{args.split}_{len(dataset)}_seed{args.seed}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
