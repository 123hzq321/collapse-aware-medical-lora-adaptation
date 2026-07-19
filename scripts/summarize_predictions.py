from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    for file_name in args.files:
        rows = [
            json.loads(line)
            for line in Path(file_name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        total = len(rows)
        correct = sum(row["gold"] == row["pred"] for row in rows)
        gold_counts = Counter(row["gold"] for row in rows)
        pred_counts = Counter(row["pred"] for row in rows)
        confusion: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            confusion[row["gold"]][row["pred"]] += 1

        print(f"\n{file_name}")
        print(f"accuracy: {correct}/{total} = {correct / total:.4f}")
        print(f"gold: {dict(gold_counts)}")
        print(f"pred: {dict(pred_counts)}")
        labels = sorted(set(gold_counts) | set(pred_counts))
        print("confusion:")
        for gold in labels:
            print(f"  {gold}: {dict(confusion[gold])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
