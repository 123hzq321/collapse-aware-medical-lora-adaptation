from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


PUBMEDQA_LABELS = ["yes", "no", "maybe"]
MCQ_LABELS = ["A", "B", "C", "D"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def infer_labels(rows: list[dict]) -> list[str]:
    seen = {str(row.get("gold", "")) for row in rows} | {str(row.get("pred", "")) for row in rows}
    if seen <= set(PUBMEDQA_LABELS + ["parse_failure"]):
        return PUBMEDQA_LABELS
    if seen <= set(MCQ_LABELS + ["parse_failure"]):
        return MCQ_LABELS
    return sorted(label for label in seen if label and label != "parse_failure")


def normalized_entropy(preds: list[str], labels: list[str]) -> float:
    counts = Counter(preds)
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


def collapse_summary(preds: list[str], labels: list[str]) -> tuple[str, float, float, bool]:
    counts = Counter(preds)
    total = len(preds)
    dominant_label, dominant_count = max(
        ((label, counts[label]) for label in labels),
        key=lambda item: item[1],
    )
    dominant_rate = dominant_count / total if total else 0.0
    entropy = normalized_entropy(preds, labels)
    missing = [label for label in labels if counts[label] == 0]
    collapse = dominant_rate >= 0.80 or len(missing) >= max(1, len(labels) // 2)
    return dominant_label, dominant_rate, entropy, collapse


def valid_pairs(gold: list[str], pred: list[str]) -> tuple[list[str], list[str]]:
    pairs = [(g, p) for g, p in zip(gold, pred) if p != "parse_failure"]
    return [g for g, _ in pairs], [p for _, p in pairs]


def summarize_pair(name: str, before_path: Path, after_path: Path) -> dict:
    before_rows = read_jsonl(before_path)
    after_rows = read_jsonl(after_path)
    if len(before_rows) != len(after_rows):
        raise ValueError(f"row count mismatch for {name}: {before_path} vs {after_path}")

    gold = [row["gold"] for row in before_rows]
    after_gold = [row["gold"] for row in after_rows]
    if gold != after_gold:
        raise ValueError(f"gold labels are not aligned for {name}")

    before_pred = [row["pred"] for row in before_rows]
    after_pred = [row["pred"] for row in after_rows]
    labels = infer_labels(before_rows + after_rows)

    before_valid_gold, before_valid_pred = valid_pairs(gold, before_pred)
    after_valid_gold, after_valid_pred = valid_pairs(gold, after_pred)
    before_acc = accuracy_score(before_valid_gold, before_valid_pred) if before_valid_gold else 0.0
    after_acc = accuracy_score(after_valid_gold, after_valid_pred) if after_valid_gold else 0.0
    before_f1 = f1_score(before_valid_gold, before_valid_pred, labels=labels, average="macro") if before_valid_gold else 0.0
    after_f1 = f1_score(after_valid_gold, after_valid_pred, labels=labels, average="macro") if after_valid_gold else 0.0

    before_correct = [g == p for g, p in zip(gold, before_pred)]
    after_correct = [g == p for g, p in zip(gold, after_pred)]
    before_only = sum(b and not a for b, a in zip(before_correct, after_correct))
    after_only = sum((not b) and a for b, a in zip(before_correct, after_correct))
    discordant = before_only + after_only
    p_value = binomtest(before_only, discordant, 0.5).pvalue if discordant else 1.0

    before_dom, before_dom_rate, before_entropy, before_collapse = collapse_summary(before_pred, labels)
    after_dom, after_dom_rate, after_entropy, after_collapse = collapse_summary(after_pred, labels)

    return {
        "pair": name,
        "n": len(gold),
        "labels": " ".join(labels),
        "before_file": str(before_path),
        "after_file": str(after_path),
        "accuracy_before": before_acc,
        "accuracy_after": after_acc,
        "delta_accuracy": after_acc - before_acc,
        "macro_f1_before": before_f1,
        "macro_f1_after": after_f1,
        "delta_macro_f1": after_f1 - before_f1,
        "before_correct_after_wrong": before_only,
        "before_wrong_after_correct": after_only,
        "mcnemar_p": p_value,
        "dominant_before": before_dom,
        "dominant_after": after_dom,
        "dominant_rate_before": before_dom_rate,
        "dominant_rate_after": after_dom_rate,
        "delta_dominant_rate": after_dom_rate - before_dom_rate,
        "entropy_before": before_entropy,
        "entropy_after": after_entropy,
        "delta_entropy": after_entropy - before_entropy,
        "collapse_before": before_collapse,
        "collapse_after": after_collapse,
    }


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = [
        "pair",
        "n",
        "accuracy_before",
        "accuracy_after",
        "delta_accuracy",
        "macro_f1_before",
        "macro_f1_after",
        "delta_macro_f1",
        "before_correct_after_wrong",
        "before_wrong_after_correct",
        "mcnemar_p",
        "dominant_rate_before",
        "dominant_rate_after",
        "delta_entropy",
        "collapse_before",
        "collapse_after",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.4g}" if header == "mcnemar_p" else f"{value:.4f}"
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("NAME", "BEFORE_JSONL", "AFTER_JSONL"),
        required=True,
    )
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args()

    rows = [
        summarize_pair(name, Path(before_path), Path(after_path))
        for name, before_path, after_path in args.pair
    ]
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
