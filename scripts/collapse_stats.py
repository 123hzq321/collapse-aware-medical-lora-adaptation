from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


LABEL_SETS = {
    "pubmedqa": ["yes", "no", "maybe"],
    "mcq": ["A", "B", "C", "D"],
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def infer_labels(rows: list[dict]) -> list[str]:
    seen = {str(row.get("gold", "")) for row in rows} | {str(row.get("pred", "")) for row in rows}
    if seen <= set(LABEL_SETS["pubmedqa"] + ["parse_failure"]):
        return LABEL_SETS["pubmedqa"]
    if seen <= set(LABEL_SETS["mcq"] + ["parse_failure"]):
        return LABEL_SETS["mcq"]
    return sorted(label for label in seen if label and label != "parse_failure")


def normalized_entropy(counts: Counter, labels: list[str]) -> float:
    total = sum(counts[label] for label in labels)
    if total <= 0 or len(labels) <= 1:
        return 0.0
    entropy = 0.0
    for label in labels:
        count = counts[label]
        if count:
            prob = count / total
            entropy -= prob * math.log(prob)
    return entropy / math.log(len(labels))


def summarize_file(path: Path) -> dict:
    rows = read_jsonl(path)
    labels = infer_labels(rows)
    gold = [row["gold"] for row in rows]
    pred = [row["pred"] for row in rows]
    valid_pairs = [(g, p) for g, p in zip(gold, pred) if p != "parse_failure"]
    valid_gold = [g for g, _ in valid_pairs]
    valid_pred = [p for _, p in valid_pairs]

    pred_counts = Counter(pred)
    gold_counts = Counter(gold)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for g, p in zip(gold, pred):
        confusion[g][p] += 1

    total = len(rows)
    valid_total = len(valid_pairs)
    dominant_label, dominant_count = ("", 0)
    if labels:
        dominant_label, dominant_count = max(
            ((label, pred_counts[label]) for label in labels),
            key=lambda item: item[1],
        )
    missing_labels = [label for label in labels if pred_counts[label] == 0]
    dominant_rate = dominant_count / total if total else 0.0
    entropy = normalized_entropy(pred_counts, labels)
    collapse_flag = bool(dominant_rate >= 0.80 or len(missing_labels) >= max(1, len(labels) // 2))

    row = {
        "file": str(path),
        "samples": total,
        "valid_samples": valid_total,
        "parse_failures": pred_counts["parse_failure"],
        "accuracy": accuracy_score(valid_gold, valid_pred) if valid_pairs else 0.0,
        "macro_f1": f1_score(valid_gold, valid_pred, labels=labels, average="macro") if valid_pairs else 0.0,
        "labels": " ".join(labels),
        "gold_counts": json.dumps(dict(gold_counts), ensure_ascii=False, sort_keys=True),
        "pred_counts": json.dumps(dict(pred_counts), ensure_ascii=False, sort_keys=True),
        "dominant_label": dominant_label,
        "dominant_rate": dominant_rate,
        "prediction_entropy_norm": entropy,
        "missing_pred_labels": " ".join(missing_labels),
        "collapse_flag": collapse_flag,
        "confusion": json.dumps(
            {label: dict(confusion[label]) for label in labels},
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    return row


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = [
        "file",
        "samples",
        "accuracy",
        "macro_f1",
        "dominant_label",
        "dominant_rate",
        "prediction_entropy_norm",
        "missing_pred_labels",
        "collapse_flag",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args()

    rows = [summarize_file(Path(file_name)) for file_name in args.files]
    print(json.dumps(rows, indent=2, ensure_ascii=False))

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if args.md_out:
        out_path = Path(args.md_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(rows, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
