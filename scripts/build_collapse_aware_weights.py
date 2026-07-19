from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def base_group(row: dict, group_by: str) -> str:
    source = str(row.get("source", "unknown"))
    task = str(row.get("task", "unknown"))
    if group_by == "global":
        return "global"
    if group_by == "source":
        return source
    if group_by == "task":
        return task
    if group_by == "source_task":
        return f"{source}:{task}"
    raise ValueError(f"unknown group_by: {group_by}")


def target_label(row: dict) -> str:
    return str(row.get("target") or row.get("gold") or "unknown")


def weight_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "mean": round(sum(values) / len(values), 6),
        "max": round(max(values), 6),
    }


def build_weighted_rows(
    rows: list[dict],
    group_by: str,
    power: float,
    min_weight: float,
    max_weight: float,
    normalize: bool,
) -> tuple[list[dict], dict]:
    label_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        label_counts[base_group(row, group_by)][target_label(row)] += 1

    raw_weights = []
    for row in rows:
        group = base_group(row, group_by)
        label = target_label(row)
        count = label_counts[group][label]
        reference = max(label_counts[group].values())
        weight = (reference / count) ** power
        raw_weights.append(weight)

    if normalize and raw_weights:
        mean_weight = sum(raw_weights) / len(raw_weights)
        raw_weights = [weight / mean_weight for weight in raw_weights]

    weighted_rows = []
    for row, raw_weight in zip(rows, raw_weights):
        group = base_group(row, group_by)
        label = target_label(row)
        clipped_weight = min(max(raw_weight, min_weight), max_weight)
        weighted = dict(row)
        weighted["sample_weight"] = round(clipped_weight, 6)
        weighted["weight_strategy"] = "collapse_aware_inverse_label_frequency"
        weighted["weight_group"] = group
        weighted["weight_label"] = label
        weighted["weight_label_count"] = label_counts[group][label]
        weighted["weight_reference_count"] = max(label_counts[group].values())
        weighted_rows.append(weighted)

    weights_by_group_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in weighted_rows:
        weights_by_group_label[row["weight_group"]][row["weight_label"]].append(row["sample_weight"])

    summary = {
        "rows": len(weighted_rows),
        "strategy": "collapse_aware_inverse_label_frequency",
        "group_by": group_by,
        "power": power,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "normalize": normalize,
        "source_counts": dict(Counter(row.get("source", "unknown") for row in weighted_rows)),
        "target_counts_by_source": {
            source: dict(Counter(row["target"] for row in weighted_rows if row.get("source") == source))
            for source in sorted({row.get("source", "unknown") for row in weighted_rows})
        },
        "sample_weight_stats": weight_stats([row["sample_weight"] for row in weighted_rows]),
        "sample_weight_stats_by_group_label": {
            group: {label: weight_stats(values) for label, values in sorted(labels.items())}
            for group, labels in sorted(weights_by_group_label.items())
        },
    }
    return weighted_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--group-by",
        default="source",
        choices=["global", "source", "task", "source_task"],
    )
    parser.add_argument("--power", type=float, default=0.5)
    parser.add_argument("--min-weight", type=float, default=0.5)
    parser.add_argument("--max-weight", type=float, default=2.0)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    weighted_rows, summary = build_weighted_rows(
        rows=rows,
        group_by=args.group_by,
        power=args.power,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        normalize=not args.no_normalize,
    )

    out_path = Path(args.out)
    write_jsonl(out_path, weighted_rows)
    summary["input"] = args.input
    summary["out"] = str(out_path)
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
