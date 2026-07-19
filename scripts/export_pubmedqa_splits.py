from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from medself.data import flatten_pubmedqa_context, load_pubmedqa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/pubmedqa")
    args = parser.parse_args()

    datasets = load_pubmedqa(seed=args.seed)
    out_dir = Path(args.out_dir)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": "qiaojin/PubMedQA",
        "config": "pqa_labeled",
        "seed": args.seed,
        "splits": {},
    }

    for split_name, split in datasets.items():
        out_path = split_dir / f"{split_name}.jsonl"
        counts = Counter()
        with out_path.open("w", encoding="utf-8") as handle:
            for index, example in enumerate(split):
                gold = example["final_decision"]
                counts[gold] += 1
                row = {
                    "id": str(example["pubid"]),
                    "index": index,
                    "split": split_name,
                    "question": example["question"],
                    "context": flatten_pubmedqa_context(example),
                    "gold": gold,
                    "long_answer": example.get("long_answer", ""),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary["splits"][split_name] = {
            "rows": len(split),
            "path": str(out_path),
            "label_counts": dict(counts),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
