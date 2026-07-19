from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = [
    "accuracy",
    "macro_f1",
    "dominant_rate",
    "prediction_entropy_norm",
]


def parse_file_name(file_name: str) -> tuple[str, str, str]:
    name = Path(file_name).stem
    if "pubmedqa" in name:
        task = "pubmedqa"
    elif "medmcqa" in name:
        task = "medmcqa"
    elif "medqa" in name:
        task = "medqa"
    else:
        task = "unknown"

    seed_match = re.search(r"seed(\d+)", name)
    seed = seed_match.group(1) if seed_match else ""

    if "base" in name:
        model_match = re.search(r"qwen\d+b", name)
        condition = (model_match.group(0) if model_match else name.split("_base")[0]) + "_base"
    elif "full45k" in name:
        condition = re.sub(r"_seed\d+.*$", "", name)
    elif "sft8k" in name or "multitask3" in name:
        condition = "qwen05b_sft8k"
    else:
        condition = re.sub(r"_(pubmedqa|medmcqa|medqa).*$", "", name)

    return task, condition, seed


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = [
        "task",
        "condition",
        "n",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "dominant_rate_mean",
        "dominant_rate_std",
        "entropy_mean",
        "entropy_std",
        "collapse_count",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-csv", required=True)
    parser.add_argument("--csv-out", default=None)
    parser.add_argument("--md-out", default=None)
    args = parser.parse_args()

    with Path(args.stats_csv).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        task, condition, seed = parse_file_name(row["file"])
        row["task"] = task
        row["condition"] = condition
        row["seed"] = seed
        grouped[(task, condition)].append(row)

    out_rows = []
    for (task, condition), items in sorted(grouped.items()):
        summary = {
            "task": task,
            "condition": condition,
            "n": len(items),
            "collapse_count": sum(str(item["collapse_flag"]).lower() == "true" for item in items),
            "seeds": " ".join(item["seed"] for item in items if item["seed"]),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items]
            mean, std = mean_std(values)
            key = "entropy" if metric == "prediction_entropy_norm" else metric
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
        out_rows.append(summary)

    print("\n".join(str(row) for row in out_rows))

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(out_rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

    if args.md_out:
        out_path = Path(args.md_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(out_rows, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
