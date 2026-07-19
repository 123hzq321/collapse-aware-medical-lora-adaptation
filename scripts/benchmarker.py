from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score


LABELS = {
    "pubmedqa": ["yes", "no", "maybe"],
    "medmcqa": ["A", "B", "C", "D"],
    "medqa": ["A", "B", "C", "D"],
}

TASK_NAMES = {
    "pubmedqa": "PubMedQA test100",
    "medmcqa": "MedMCQA val1000",
    "medqa": "MedQA test1273",
}


@dataclass(frozen=True)
class PredictionSpec:
    task: str
    model: str
    condition: str
    seed: str
    path: str
    family: str


PREDICTIONS = [
    PredictionSpec("pubmedqa", "qwen05b", "base", "", "runs/pubmedqa_qwen05b_base_test100.jsonl", "frozen_baseline"),
    PredictionSpec("medmcqa", "qwen05b", "base", "", "runs/qwen05b_base_medmcqa_val1000.jsonl", "frozen_baseline"),
    PredictionSpec("medqa", "qwen05b", "base", "", "runs/qwen05b_base_medqa_test1273.jsonl", "frozen_baseline"),
    PredictionSpec("pubmedqa", "qwen05b", "full45k", "42", "runs/qwen05b_full45k_seed42_pubmedqa_test100.jsonl", "self_update"),
    PredictionSpec("pubmedqa", "qwen05b", "full45k", "43", "runs/qwen05b_full45k_seed43_pubmedqa_test100.jsonl", "self_update"),
    PredictionSpec("pubmedqa", "qwen05b", "full45k", "44", "runs/qwen05b_full45k_seed44_pubmedqa_test100.jsonl", "self_update"),
    PredictionSpec("medmcqa", "qwen05b", "full45k", "42", "runs/qwen05b_full45k_seed42_medmcqa_val1000.jsonl", "self_update"),
    PredictionSpec("medmcqa", "qwen05b", "full45k", "43", "runs/qwen05b_full45k_seed43_medmcqa_val1000.jsonl", "self_update"),
    PredictionSpec("medmcqa", "qwen05b", "full45k", "44", "runs/qwen05b_full45k_seed44_medmcqa_val1000.jsonl", "self_update"),
    PredictionSpec("medqa", "qwen05b", "full45k", "42", "runs/qwen05b_full45k_seed42_medqa_test1273.jsonl", "self_update"),
    PredictionSpec("medqa", "qwen05b", "full45k", "43", "runs/qwen05b_full45k_seed43_medqa_test1273.jsonl", "self_update"),
    PredictionSpec("medqa", "qwen05b", "full45k", "44", "runs/qwen05b_full45k_seed44_medqa_test1273.jsonl", "self_update"),
    PredictionSpec("pubmedqa", "qwen15b", "base", "", "runs/qwen15b_base_pubmedqa_test100.jsonl", "frozen_baseline"),
    PredictionSpec("medmcqa", "qwen15b", "base", "", "runs/qwen15b_base_medmcqa_val1000.jsonl", "frozen_baseline"),
    PredictionSpec("medqa", "qwen15b", "base", "", "runs/qwen15b_base_medqa_test1273.jsonl", "frozen_baseline"),
    PredictionSpec("pubmedqa", "qwen15b", "full45k", "42", "runs/qwen15b_full45k_seed42_pubmedqa_test100.jsonl", "self_update"),
    PredictionSpec("pubmedqa", "qwen15b", "full45k", "43", "runs/qwen15b_full45k_seed43_pubmedqa_test100.jsonl", "self_update"),
    PredictionSpec("pubmedqa", "qwen15b", "full45k", "44", "runs/qwen15b_full45k_seed44_pubmedqa_test100.jsonl", "self_update"),
    PredictionSpec("medmcqa", "qwen15b", "full45k", "42", "runs/qwen15b_full45k_seed42_medmcqa_val1000.jsonl", "self_update"),
    PredictionSpec("medmcqa", "qwen15b", "full45k", "43", "runs/qwen15b_full45k_seed43_medmcqa_val1000.jsonl", "self_update"),
    PredictionSpec("medmcqa", "qwen15b", "full45k", "44", "runs/qwen15b_full45k_seed44_medmcqa_val1000.jsonl", "self_update"),
    PredictionSpec("medqa", "qwen15b", "full45k", "42", "runs/qwen15b_full45k_seed42_medqa_test1273.jsonl", "self_update"),
    PredictionSpec("medqa", "qwen15b", "full45k", "43", "runs/qwen15b_full45k_seed43_medqa_test1273.jsonl", "self_update"),
    PredictionSpec("medqa", "qwen15b", "full45k", "44", "runs/qwen15b_full45k_seed44_medqa_test1273.jsonl", "self_update"),
    PredictionSpec("pubmedqa", "qwen3b", "base", "", "runs/qwen3b_base_pubmedqa_test100.jsonl", "frozen_baseline"),
    PredictionSpec("medmcqa", "qwen3b", "base", "", "runs/qwen3b_base_medmcqa_val1000.jsonl", "frozen_baseline"),
    PredictionSpec("medqa", "qwen3b", "base", "", "runs/qwen3b_base_medqa_test1273.jsonl", "frozen_baseline"),
    PredictionSpec("pubmedqa", "qwen3b", "full45k", "42", "runs/qwen3b_full45k_seed42_pubmedqa_test100.jsonl", "self_update"),
    PredictionSpec("pubmedqa", "qwen3b", "full45k", "43", "runs/qwen3b_full45k_seed43_pubmedqa_test100.jsonl", "self_update"),
    PredictionSpec("pubmedqa", "qwen3b", "full45k", "44", "runs/qwen3b_full45k_seed44_pubmedqa_test100.jsonl", "self_update"),
    PredictionSpec("medmcqa", "qwen3b", "full45k", "42", "runs/qwen3b_full45k_seed42_medmcqa_val1000.jsonl", "self_update"),
    PredictionSpec("medmcqa", "qwen3b", "full45k", "43", "runs/qwen3b_full45k_seed43_medmcqa_val1000.jsonl", "self_update"),
    PredictionSpec("medmcqa", "qwen3b", "full45k", "44", "runs/qwen3b_full45k_seed44_medmcqa_val1000.jsonl", "self_update"),
    PredictionSpec("medqa", "qwen3b", "full45k", "42", "runs/qwen3b_full45k_seed42_medqa_test1273.jsonl", "self_update"),
    PredictionSpec("medqa", "qwen3b", "full45k", "43", "runs/qwen3b_full45k_seed43_medqa_test1273.jsonl", "self_update"),
    PredictionSpec("medqa", "qwen3b", "full45k", "44", "runs/qwen3b_full45k_seed44_medqa_test1273.jsonl", "self_update"),
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def entropy_norm(preds: list[str], labels: list[str]) -> float:
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


def metric_row(
    *,
    task: str,
    family: str,
    model: str,
    condition: str,
    seed: str,
    gold: list[str],
    pred: list[str],
    source_file: str,
) -> dict:
    labels = LABELS[task]
    pred_counts = Counter(pred)
    gold_counts = Counter(gold)
    dominant_label, dominant_count = max(
        ((label, pred_counts[label]) for label in labels),
        key=lambda item: item[1],
    )
    dominant_rate = dominant_count / len(pred) if pred else 0.0
    missing_labels = [label for label in labels if pred_counts[label] == 0]
    collapse = dominant_rate >= 0.80 or len(missing_labels) >= max(1, len(labels) // 2)
    valid_pairs = [(g, p) for g, p in zip(gold, pred) if p != "parse_failure"]
    valid_gold = [g for g, _ in valid_pairs]
    valid_pred = [p for _, p in valid_pairs]
    return {
        "task": task,
        "task_name": TASK_NAMES[task],
        "family": family,
        "model": model,
        "condition": condition,
        "seed": seed,
        "n": len(gold),
        "accuracy": accuracy_score(valid_gold, valid_pred) if valid_pairs else "",
        "macro_f1": f1_score(valid_gold, valid_pred, labels=labels, average="macro") if valid_pairs else "",
        "dominant_label": dominant_label,
        "dominant_rate": dominant_rate,
        "entropy": entropy_norm(pred, labels),
        "collapse_flag": collapse,
        "gold_counts": json.dumps(dict(gold_counts), ensure_ascii=False, sort_keys=True),
        "pred_counts": json.dumps(dict(pred_counts), ensure_ascii=False, sort_keys=True),
        "expected_accuracy": "",
        "source_file": source_file,
    }


def expected_row(
    *,
    task: str,
    family: str,
    model: str,
    condition: str,
    gold: list[str],
    expected_accuracy: float,
    source_file: str,
) -> dict:
    return {
        "task": task,
        "task_name": TASK_NAMES[task],
        "family": family,
        "model": model,
        "condition": condition,
        "seed": "",
        "n": len(gold),
        "accuracy": "",
        "macro_f1": "",
        "dominant_label": "",
        "dominant_rate": "",
        "entropy": "",
        "collapse_flag": "",
        "gold_counts": json.dumps(dict(Counter(gold)), ensure_ascii=False, sort_keys=True),
        "pred_counts": "",
        "expected_accuracy": expected_accuracy,
        "source_file": source_file,
    }


def load_gold_by_task(root: Path) -> dict[str, tuple[list[str], str]]:
    gold_files = {
        "pubmedqa": "runs/pubmedqa_qwen05b_base_test100.jsonl",
        "medmcqa": "runs/qwen05b_base_medmcqa_val1000.jsonl",
        "medqa": "runs/qwen05b_base_medqa_test1273.jsonl",
    }
    gold_by_task = {}
    for task, file_name in gold_files.items():
        rows = read_jsonl(root / file_name)
        gold_by_task[task] = ([row["gold"] for row in rows], file_name)
    return gold_by_task


def trivial_baselines(root: Path) -> list[dict]:
    rows = []
    for task, (gold, file_name) in load_gold_by_task(root).items():
        labels = LABELS[task]
        counts = Counter(gold)
        majority_label = max(labels, key=lambda label: counts[label])
        majority_pred = [majority_label] * len(gold)
        rows.append(
            metric_row(
                task=task,
                family="label_baseline",
                model="gold_majority",
                condition=f"always_{majority_label}",
                seed="",
                gold=gold,
                pred=majority_pred,
                source_file=file_name,
            )
        )
        rows.append(
            expected_row(
                task=task,
                family="label_baseline",
                model="uniform_random",
                condition="expected",
                gold=gold,
                expected_accuracy=1 / len(labels),
                source_file=file_name,
            )
        )
        total = len(gold)
        prior_expected = sum((counts[label] / total) ** 2 for label in labels)
        rows.append(
            expected_row(
                task=task,
                family="label_baseline",
                model="label_prior_random",
                condition="expected",
                gold=gold,
                expected_accuracy=prior_expected,
                source_file=file_name,
            )
        )
    return rows


def model_rows(root: Path) -> list[dict]:
    rows = []
    for spec in PREDICTIONS:
        path = root / spec.path
        if not path.exists():
            continue
        data = read_jsonl(path)
        rows.append(
            metric_row(
                task=spec.task,
                family=spec.family,
                model=spec.model,
                condition=spec.condition,
                seed=spec.seed,
                gold=[row["gold"] for row in data],
                pred=[row["pred"] for row in data],
                source_file=spec.path,
            )
        )
    return rows


def grouped_mean_rows(rows: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["family"] not in {"frozen_baseline", "self_update"}:
            continue
        grouped[(row["task"], row["family"], row["model"], row["condition"])].append(row)

    out = []
    for (task, family, model, condition), items in sorted(grouped.items()):
        summary = {
            "task": task,
            "task_name": TASK_NAMES[task],
            "family": family,
            "model": model,
            "condition": condition,
            "n_runs": len(items),
            "seeds": " ".join(item["seed"] for item in items if item["seed"]),
            "collapse_count": sum(str(item["collapse_flag"]).lower() == "true" for item in items),
        }
        for metric in ["accuracy", "macro_f1", "dominant_rate", "entropy"]:
            values = [float(item[metric]) for item in items if item[metric] != ""]
            if not values:
                summary[f"{metric}_mean"] = ""
                summary[f"{metric}_std"] = ""
            elif len(values) == 1:
                summary[f"{metric}_mean"] = values[0]
                summary[f"{metric}_std"] = 0.0
            else:
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_std"] = statistics.stdev(values)
        out.append(summary)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(rows: list[dict], path: Path, headers: list[str]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header, "")).replace("|", "/") for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    *,
    baseline_rows: list[dict],
    aggregate_rows: list[dict],
    all_rows: list[dict],
    path: Path,
) -> None:
    best_by_task = {}
    for task in LABELS:
        task_rows = [
            row for row in aggregate_rows
            if row["task"] == task and row["family"] == "self_update"
        ]
        if task_rows:
            best_by_task[task] = max(task_rows, key=lambda row: float(row["accuracy_mean"]))

    lines = [
        "# BENCHMARKER Current Report",
        "",
        "Scope: PubMedQA test100, MedMCQA val1000, and MedQA test1273. Metrics are accuracy, macro-F1, dominant-label rate, normalized prediction entropy, and collapse flag.",
        "",
        "## Non-Model Baselines",
        "",
        "| task | baseline | condition | accuracy | macro-F1 | expected accuracy | collapse |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in baseline_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task_name"],
                    row["model"],
                    row["condition"],
                    fmt(row["accuracy"]),
                    fmt(row["macro_f1"]),
                    fmt(row["expected_accuracy"]),
                    fmt(row["collapse_flag"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Best Current Self-Update Runs",
            "",
            "| task | model | condition | seeds/runs | accuracy | macro-F1 | dominant rate | entropy | collapse |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task in LABELS:
        row = best_by_task.get(task)
        if not row:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task_name"],
                    row["model"],
                    row["condition"],
                    row["seeds"] or str(row["n_runs"]),
                    fmt(row["accuracy_mean"]),
                    fmt(row["macro_f1_mean"]),
                    fmt(row["dominant_rate_mean"]),
                    fmt(row["entropy_mean"]),
                    f"{row['collapse_count']}/{row['n_runs']}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Frozen vs Full45k Summary",
            "",
            "| task | model | frozen acc | full45k acc | delta | frozen collapse | full45k collapse |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    by_key = {(row["task"], row["model"], row["condition"]): row for row in aggregate_rows}
    for task in LABELS:
        for model in ["qwen05b", "qwen15b", "qwen3b"]:
            base = by_key.get((task, model, "base"))
            full = by_key.get((task, model, "full45k"))
            if not base or not full:
                continue
            delta = float(full["accuracy_mean"]) - float(base["accuracy_mean"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        TASK_NAMES[task],
                        model,
                        fmt(base["accuracy_mean"]),
                        fmt(full["accuracy_mean"]),
                        fmt(delta),
                        f"{base['collapse_count']}/{base['n_runs']}",
                        f"{full['collapse_count']}/{full['n_runs']}",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Artifact Index",
            "",
            "- `runs/benchmarker_baselines_current.csv`: non-model baselines.",
            "- `runs/benchmarker_predictions_current.csv`: per-run model metrics.",
            "- `runs/benchmarker_aggregate_current.csv`: grouped model metrics.",
            "- `runs/benchmarker_current.md`: this report.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-prefix", default="runs/benchmarker_current")
    args = parser.parse_args()

    root = Path(args.root)
    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = root / out_prefix

    baseline_rows = trivial_baselines(root)
    prediction_rows = model_rows(root)
    aggregate_rows = grouped_mean_rows(prediction_rows)
    all_rows = baseline_rows + prediction_rows

    write_csv(baseline_rows, out_prefix.with_name("benchmarker_baselines_current.csv"))
    write_csv(prediction_rows, out_prefix.with_name("benchmarker_predictions_current.csv"))
    write_csv(aggregate_rows, out_prefix.with_name("benchmarker_aggregate_current.csv"))
    write_csv(all_rows, out_prefix.with_name("benchmarker_all_current.csv"))

    write_markdown_table(
        baseline_rows,
        out_prefix.with_name("benchmarker_baselines_current.md"),
        ["task_name", "family", "model", "condition", "n", "accuracy", "macro_f1", "expected_accuracy", "collapse_flag"],
    )
    write_markdown_table(
        aggregate_rows,
        out_prefix.with_name("benchmarker_aggregate_current.md"),
        [
            "task_name",
            "family",
            "model",
            "condition",
            "n_runs",
            "seeds",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "dominant_rate_mean",
            "entropy_mean",
            "collapse_count",
        ],
    )
    write_report(
        baseline_rows=baseline_rows,
        aggregate_rows=aggregate_rows,
        all_rows=all_rows,
        path=out_prefix.with_suffix(".md"),
    )
    print(f"wrote benchmarker artifacts with prefix: {out_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
