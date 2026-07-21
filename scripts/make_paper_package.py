from __future__ import annotations

import argparse
import csv
from pathlib import Path


TASK_ORDER = ["pubmedqa", "medmcqa", "medqa"]
TASK_LABEL = {
    "pubmedqa": "PubMedQA test100",
    "medmcqa": "MedMCQA val1000",
    "medqa": "MedQA test1273",
}
MODEL_ORDER = ["qwen05b", "qwen15b", "qwen3b"]
MODEL_LABEL = {
    "qwen05b": "Qwen2.5-0.5B",
    "qwen15b": "Qwen2.5-1.5B",
    "qwen3b": "Qwen2.5-3B",
}
PAIR_LABEL = {
    "qwen05b_pubmedqa_s42": "0.5B PubMedQA seed42",
    "qwen05b_pubmedqa_s43": "0.5B PubMedQA seed43",
    "qwen05b_pubmedqa_s44": "0.5B PubMedQA seed44",
    "qwen05b_medmcqa_s42": "0.5B MedMCQA seed42",
    "qwen05b_medmcqa_s43": "0.5B MedMCQA seed43",
    "qwen05b_medmcqa_s44": "0.5B MedMCQA seed44",
    "qwen05b_medqa_s42": "0.5B MedQA seed42",
    "qwen05b_medqa_s43": "0.5B MedQA seed43",
    "qwen05b_medqa_s44": "0.5B MedQA seed44",
    "qwen15b_pubmedqa_s42": "1.5B PubMedQA seed42",
    "qwen15b_pubmedqa_s43": "1.5B PubMedQA seed43",
    "qwen15b_pubmedqa_s44": "1.5B PubMedQA seed44",
    "qwen15b_medmcqa_s42": "1.5B MedMCQA seed42",
    "qwen15b_medmcqa_s43": "1.5B MedMCQA seed43",
    "qwen15b_medmcqa_s44": "1.5B MedMCQA seed44",
    "qwen15b_medqa_s42": "1.5B MedQA seed42",
    "qwen15b_medqa_s43": "1.5B MedQA seed43",
    "qwen15b_medqa_s44": "1.5B MedQA seed44",
    "qwen3b_pubmedqa_s42": "3B PubMedQA seed42",
    "qwen3b_pubmedqa_s43": "3B PubMedQA seed43",
    "qwen3b_pubmedqa_s44": "3B PubMedQA seed44",
    "qwen3b_medmcqa_s42": "3B MedMCQA seed42",
    "qwen3b_medmcqa_s43": "3B MedMCQA seed43",
    "qwen3b_medmcqa_s44": "3B MedMCQA seed44",
    "qwen3b_medqa_s42": "3B MedQA seed42",
    "qwen3b_medqa_s43": "3B MedQA seed43",
    "qwen3b_medqa_s44": "3B MedQA seed44",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f4(value: str | float) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.4f}"


def pm(mean: str, std: str, n_runs: str) -> str:
    if mean == "":
        return ""
    if int(n_runs) > 1:
        return f"{float(mean):.3f} +/- {float(std):.3f}"
    return f"{float(mean):.3f}"


def p_value(value: str) -> str:
    val = float(value)
    if val < 1e-4:
        return f"{val:.2e}"
    return f"{val:.4f}"


def markdown_table(rows: list[dict], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "/") for header in headers) + " |")
    return "\n".join(lines)


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def latex_table(caption: str, label: str, rows: list[dict], headers: list[str]) -> str:
    alignment = "l" * len(headers)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(latex_escape(header) for header in headers) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(header, "")) for header in headers) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def build_main_rows(aggregate_rows: list[dict]) -> list[dict]:
    by_key = {(row["task"], row["model"], row["condition"]): row for row in aggregate_rows}
    out = []
    for task in TASK_ORDER:
        for model in MODEL_ORDER:
            base = by_key.get((task, model, "base"))
            full = by_key.get((task, model, "full45k"))
            if not base or not full:
                continue
            delta = float(full["accuracy_mean"]) - float(base["accuracy_mean"])
            out.append(
                {
                    "Task": TASK_LABEL[task],
                    "Model": MODEL_LABEL[model],
                    "Frozen Acc.": pm(base["accuracy_mean"], base["accuracy_std"], base["n_runs"]),
                    "Full45k Acc.": pm(full["accuracy_mean"], full["accuracy_std"], full["n_runs"]),
                    "Delta Acc.": f"{delta:+.3f}",
                    "Full45k Macro-F1": pm(full["macro_f1_mean"], full["macro_f1_std"], full["n_runs"]),
                    "Frozen Collapse": f"{base['collapse_count']}/{base['n_runs']}",
                    "Full45k Collapse": f"{full['collapse_count']}/{full['n_runs']}",
                }
            )
    return out


def build_baseline_rows(baseline_rows: list[dict]) -> list[dict]:
    out = []
    for row in baseline_rows:
        if row["model"] == "gold_majority":
            score = f4(row["accuracy"])
            f1 = f4(row["macro_f1"])
        else:
            score = f"{f4(row['expected_accuracy'])} expected"
            f1 = ""
        out.append(
            {
                "Task": row["task_name"],
                "Baseline": row["model"],
                "Condition": row["condition"],
                "Accuracy": score,
                "Macro-F1": f1,
                "Collapse": str(row["collapse_flag"]),
            }
        )
    return out


def build_significance_rows(paired_rows: list[dict]) -> list[dict]:
    rows = []
    for row in paired_rows:
        if row["pair"] not in PAIR_LABEL:
            continue
        rows.append(
            {
                "Comparison": PAIR_LABEL[row["pair"]],
                "Before Acc.": f4(row["accuracy_before"]),
                "After Acc.": f4(row["accuracy_after"]),
                "Delta Acc.": f"{float(row['delta_accuracy']):+.4f}",
                "McNemar p": p_value(row["mcnemar_p"]),
                "Before Collapse": row["collapse_before"],
                "After Collapse": row["collapse_after"],
            }
        )
    return rows


def write_results_section(
    path: Path,
    main_rows: list[dict],
    baseline_rows: list[dict],
    significance_rows: list[dict],
) -> None:
    q05_pub = next(row for row in main_rows if row["Task"].startswith("PubMedQA") and row["Model"] == "Qwen2.5-0.5B")
    q15_pub = next(row for row in main_rows if row["Task"].startswith("PubMedQA") and row["Model"] == "Qwen2.5-1.5B")
    q3_medqa = next(row for row in main_rows if row["Task"].startswith("MedQA") and row["Model"] == "Qwen2.5-3B")
    q3_medmcqa = next(row for row in main_rows if row["Task"].startswith("MedMCQA") and row["Model"] == "Qwen2.5-3B")
    text = f"""# Paper-Ready Results Draft

## Chinese Results Draft

我们首先构建了非模型标签基线。PubMedQA 的 majority baseline 为 always-yes，accuracy 为 0.560，但 macro-F1 仅为 0.239，且被 collapse 指标标记为坍缩；MedMCQA 与 MedQA 的 majority baseline 分别为 always-A，accuracy 为 0.323 和 0.277，macro-F1 分别为 0.122 和 0.109，同样属于单标签坍缩。因此，单独报告 accuracy 会高估标签偏置方法的有效性，后续实验同时报告 macro-F1、dominant-rate、entropy 和 collapse flag。

在 0.5B 模型上，full45k LoRA adaptation 在三个 seed 上稳定改善三项医学任务。PubMedQA 从 frozen baseline 的 {q05_pub['Frozen Acc.']} 提升到 {q05_pub['Full45k Acc.']}，并将 collapse 从 {q05_pub['Frozen Collapse']} 降到 {q05_pub['Full45k Collapse']}。MedMCQA 和 MedQA 也分别获得稳定提升，说明 full45k 不只是改变 dominant label，而是改善了多类覆盖。

在更大模型上，1.5B 与 3B 的 frozen baseline 并不天然免疫 collapse。尤其 3B frozen baseline 在 PubMedQA 上仍出现 maybe-dominant collapse。full45k 后，3B 在 MedMCQA 达到 {q3_medmcqa['Full45k Acc.']} accuracy / {q3_medmcqa['Full45k Macro-F1']} macro-F1，在 MedQA 达到 {q3_medqa['Full45k Acc.']} accuracy / {q3_medqa['Full45k Macro-F1']} macro-F1；PubMedQA 的最佳结果来自 1.5B full45k，达到 {q15_pub['Full45k Acc.']} accuracy / {q15_pub['Full45k Macro-F1']} macro-F1。

McNemar paired tests 进一步支持这些改善并非简单波动。0.5B full45k 在三任务三 seed 上均显著改善；1.5B full45k 在 MedMCQA 和 MedQA 的三个 seed 上均显著改善，PubMedQA 则稳定解除 collapse 但 accuracy 的配对检验尚未达到显著；3B full45k 在三任务三 seed 上也均达到显著提升，同时解除 PubMedQA collapse。

## English Results Draft

We first report non-model label baselines. The majority baseline reaches 0.560 accuracy on PubMedQA by always predicting yes, but its macro-F1 is only 0.239 and it is flagged as collapsed. Similarly, majority baselines on MedMCQA and MedQA obtain 0.323 and 0.277 accuracy, with macro-F1 scores of 0.122 and 0.109. These results show that accuracy alone can overstate label-biased behavior; we therefore report macro-F1, dominant-label rate, normalized prediction entropy, and collapse flags throughout.

For the 0.5B model, full45k LoRA adaptation consistently improves all three medical QA tasks across three seeds. On PubMedQA, accuracy improves from {q05_pub['Frozen Acc.']} to {q05_pub['Full45k Acc.']}, while collapse is reduced from {q05_pub['Frozen Collapse']} to {q05_pub['Full45k Collapse']}. MedMCQA and MedQA also improve, indicating that the adaptation improves multi-class coverage rather than merely shifting the dominant label.

Larger models are not automatically immune to collapse. In particular, the 3B frozen baseline still collapses on PubMedQA with a maybe-dominant prediction pattern. After full45k LoRA adaptation, the 3B model reaches {q3_medmcqa['Full45k Acc.']} accuracy / {q3_medmcqa['Full45k Macro-F1']} macro-F1 on MedMCQA and {q3_medqa['Full45k Acc.']} accuracy / {q3_medqa['Full45k Macro-F1']} macro-F1 on MedQA. The best PubMedQA score is obtained by the 1.5B full45k model, with {q15_pub['Full45k Acc.']} accuracy / {q15_pub['Full45k Macro-F1']} macro-F1.

Paired McNemar tests support that these gains are not only aggregate fluctuations. The 0.5B full45k model significantly improves over its frozen baseline across all three tasks and seeds. For the 1.5B model, MedMCQA and MedQA improve significantly across all three seeds, while PubMedQA consistently removes collapse without reaching paired-test significance for accuracy. The 3B full45k model also significantly improves over the 3B frozen baseline across all three tasks and seeds, while removing PubMedQA collapse.

## Main Result Table

{markdown_table(main_rows, list(main_rows[0].keys()))}

## Non-Model Baseline Table

{markdown_table(baseline_rows, list(baseline_rows[0].keys()))}

## Paired Significance Table

{markdown_table(significance_rows, list(significance_rows[0].keys()))}
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--outputs-dir", default="../../outputs")
    args = parser.parse_args()

    root = Path(args.root)
    outputs_dir = (root / args.outputs_dir).resolve()
    runs_dir = root / "runs"

    aggregate_rows = read_csv(runs_dir / "benchmarker_aggregate_current.csv")
    baseline_rows_raw = read_csv(runs_dir / "benchmarker_baselines_current.csv")
    paired_rows = read_csv(runs_dir / "paired_stats_current.csv")

    main_rows = build_main_rows(aggregate_rows)
    baseline_rows = build_baseline_rows(baseline_rows_raw)
    significance_rows = build_significance_rows(paired_rows)

    write_csv(main_rows, runs_dir / "paper_main_results_current.csv")
    write_csv(baseline_rows, runs_dir / "paper_non_model_baselines_current.csv")
    write_csv(significance_rows, runs_dir / "paper_paired_significance_current.csv")

    (runs_dir / "paper_main_results_current.md").write_text(
        markdown_table(main_rows, list(main_rows[0].keys())) + "\n",
        encoding="utf-8",
    )
    (runs_dir / "paper_non_model_baselines_current.md").write_text(
        markdown_table(baseline_rows, list(baseline_rows[0].keys())) + "\n",
        encoding="utf-8",
    )
    (runs_dir / "paper_paired_significance_current.md").write_text(
        markdown_table(significance_rows, list(significance_rows[0].keys())) + "\n",
        encoding="utf-8",
    )

    latex = "\n\n".join(
        [
            latex_table("Main frozen-vs-full45k benchmark results.", "tab:main_results", main_rows, list(main_rows[0].keys())),
            latex_table("Non-model label baselines.", "tab:label_baselines", baseline_rows, list(baseline_rows[0].keys())),
            latex_table("Paired McNemar significance tests.", "tab:paired_tests", significance_rows, list(significance_rows[0].keys())),
        ]
    )
    (outputs_dir / "paper_tables.tex").write_text(latex + "\n", encoding="utf-8")
    write_results_section(
        outputs_dir / "paper_results_section.md",
        main_rows=main_rows,
        baseline_rows=baseline_rows,
        significance_rows=significance_rows,
    )
    print(f"wrote paper package under {runs_dir} and {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
