from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


LABELS = {
    "pubmedqa": ["yes", "no", "maybe"],
    "medmcqa": ["A", "B", "C", "D"],
    "medqa": ["A", "B", "C", "D"],
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def per_class_rows(spec: dict, rows: list[dict]) -> list[dict]:
    labels = LABELS[spec["task"]]
    gold = [row["gold"] for row in rows]
    pred = [row["pred"] for row in rows]
    out = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        out.append(
            {
                "task": spec["task"],
                "model": spec["model"],
                "condition": spec["condition"],
                "seed": spec["seed"],
                "family": spec["family"],
                "label": label,
                "support": sum(g == label for g in gold),
                "pred_count": sum(p == label for p in pred),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "source_file": spec["source_file"],
            }
        )
    return out


def confusion_rows(spec: dict, rows: list[dict]) -> list[dict]:
    labels = LABELS[spec["task"]]
    pred_labels = labels + (["parse_failure"] if any(row["pred"] == "parse_failure" for row in rows) else [])
    counts: dict[tuple[str, str], int] = Counter((row["gold"], row["pred"]) for row in rows)
    out = []
    for gold_label in labels:
        for pred_label in pred_labels:
            out.append(
                {
                    "task": spec["task"],
                    "model": spec["model"],
                    "condition": spec["condition"],
                    "seed": spec["seed"],
                    "family": spec["family"],
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "count": counts[(gold_label, pred_label)],
                    "source_file": spec["source_file"],
                }
            )
    return out


def prediction_count_rows(spec: dict, rows: list[dict]) -> list[dict]:
    labels = LABELS[spec["task"]]
    pred_counts = Counter(row["pred"] for row in rows)
    gold_counts = Counter(row["gold"] for row in rows)
    out = []
    for label in labels + (["parse_failure"] if pred_counts["parse_failure"] else []):
        out.append(
            {
                "task": spec["task"],
                "model": spec["model"],
                "condition": spec["condition"],
                "seed": spec["seed"],
                "family": spec["family"],
                "label": label,
                "gold_count": gold_counts[label],
                "pred_count": pred_counts[label],
                "pred_rate": safe_div(pred_counts[label], len(rows)),
                "source_file": spec["source_file"],
            }
        )
    return out


def aggregate_per_class(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["task"],
                row["model"],
                row["condition"],
                row["family"],
                row["label"],
                str(row["support"]),
            )
        ].append(row)

    out = []
    for (task, model, condition, family, label, support), items in sorted(grouped.items()):
        summary = {
            "task": task,
            "model": model,
            "condition": condition,
            "family": family,
            "label": label,
            "support": support,
            "n_runs": len(items),
            "seeds": " ".join(str(item["seed"]) for item in items if item["seed"]),
        }
        for metric in ["pred_count", "precision", "recall", "f1"]:
            values = [float(item[metric]) for item in items]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        out.append(summary)
    return out


def write_per_class_md(rows: list[dict], path: Path) -> None:
    headers = [
        "task",
        "model",
        "condition",
        "seed",
        "label",
        "support",
        "pred_count",
        "precision",
        "recall",
        "f1",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_class_aggregate_md(rows: list[dict], path: Path) -> None:
    headers = [
        "task",
        "model",
        "condition",
        "label",
        "support",
        "n_runs",
        "seeds",
        "pred_count_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prediction_counts_md(rows: list[dict], path: Path) -> None:
    headers = ["task", "model", "condition", "seed", "label", "gold_count", "pred_count", "pred_rate"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confusion_md(rows: list[dict], path: Path) -> None:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["model"], row["condition"], str(row["seed"]))].append(row)

    lines = ["# Confusion Matrices", ""]
    for key, items in sorted(grouped.items()):
        task, model, condition, seed = key
        gold_labels = LABELS[task]
        pred_labels = sorted({row["pred_label"] for row in items}, key=lambda label: LABELS[task].index(label) if label in LABELS[task] else 99)
        counts = {(row["gold_label"], row["pred_label"]): row["count"] for row in items}
        seed_text = f" seed {seed}" if seed else ""
        lines.extend(
            [
                f"## {task} / {model} / {condition}{seed_text}",
                "",
                "| gold \\ pred | " + " | ".join(pred_labels) + " |",
                "|" + "|".join(["---"] * (len(pred_labels) + 1)) + "|",
            ]
        )
        for gold_label in gold_labels:
            cells = [str(counts.get((gold_label, pred_label), 0)) for pred_label in pred_labels]
            lines.append("| " + gold_label + " | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--prediction-index", default="runs/benchmarker_predictions_current.csv")
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()

    root = Path(args.root)
    prediction_index = root / args.prediction_index
    out_dir = root / args.out_dir
    specs = read_csv(prediction_index)

    per_class: list[dict] = []
    confusion: list[dict] = []
    prediction_counts: list[dict] = []
    for spec in specs:
        data_path = root / spec["source_file"]
        if not data_path.exists():
            continue
        rows = read_jsonl(data_path)
        per_class.extend(per_class_rows(spec, rows))
        confusion.extend(confusion_rows(spec, rows))
        prediction_counts.extend(prediction_count_rows(spec, rows))

    per_class_aggregate = aggregate_per_class(per_class)

    write_csv(per_class, out_dir / "per_class_metrics_current.csv")
    write_csv(per_class_aggregate, out_dir / "per_class_metrics_aggregate_current.csv")
    write_csv(confusion, out_dir / "confusion_matrices_current.csv")
    write_csv(prediction_counts, out_dir / "prediction_counts_current.csv")

    write_per_class_md(per_class, out_dir / "per_class_metrics_current.md")
    write_per_class_aggregate_md(per_class_aggregate, out_dir / "per_class_metrics_aggregate_current.md")
    write_confusion_md(confusion, out_dir / "confusion_matrices_current.md")
    write_prediction_counts_md(prediction_counts, out_dir / "prediction_counts_current.md")

    print(f"wrote detailed diagnostic tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
