from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


CHOICE_LABELS = ["A", "B", "C", "D"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/medqa")
    args = parser.parse_args()

    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")[args.split]
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size_tag = len(dataset) if args.max_samples is not None else "full"
    out_path = out_dir / f"{args.split}_{size_tag}_seed{args.seed}.jsonl"

    counts = Counter()
    meta_info = Counter()
    with out_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(dataset):
            answer_label = str(example["answer_idx"]).strip().upper()
            if answer_label not in CHOICE_LABELS:
                continue

            choices = {
                label: str(example["options"][label])
                for label in CHOICE_LABELS
                if label in example["options"]
            }
            if len(choices) != 4:
                continue

            counts[answer_label] += 1
            meta_info[example.get("meta_info") or ""] += 1
            row = {
                "id": f"{args.split}-{index}",
                "index": index,
                "source_split": args.split,
                "question": example["question"],
                "choices": choices,
                "gold": answer_label,
                "answer": example.get("answer", ""),
                "meta_info": example.get("meta_info", ""),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "GBaker/MedQA-USMLE-4-options",
        "split": args.split,
        "rows": sum(counts.values()),
        "seed": args.seed,
        "path": str(out_path),
        "label_counts": dict(counts),
        "meta_info_counts": dict(meta_info),
    }
    summary_path = out_dir / f"{args.split}_{size_tag}_seed{args.seed}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
