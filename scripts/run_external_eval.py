from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from medself.prompts import CHOICE_LABELS, build_medmcqa_prompt, build_pubmedqa_prompt


LABELS_BY_TASK_TYPE = {
    "yes_no_maybe": ["yes", "no", "maybe"],
    "yes_no": ["yes", "no"],
    "mcq4": ["A", "B", "C", "D"],
    "nli2": ["entailment", "not_entailment"],
    "nli3": ["entailment", "contradiction", "neutral"],
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_chat(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}"


def prompt_for_row(tokenizer, row: dict, template_id: int = 0) -> str:
    task_type = row["task_type"]
    if task_type == "yes_no_maybe":
        return build_pubmedqa_prompt(tokenizer, row["question"], row.get("context", ""), template_id)
    if task_type == "mcq4":
        choices = row["choices"]
        if isinstance(choices, list):
            choices = {label: choices[index] for index, label in enumerate(CHOICE_LABELS)}
        return build_medmcqa_prompt(tokenizer, row["question"], choices)
    if task_type == "yes_no":
        user = f"""Answer the question using the provided context.

Context:
{row.get("context", "")}

Question:
{row["question"]}

Return exactly one label: yes or no.
Answer:"""
        return render_chat(tokenizer, "You are a question-answering classifier. Return only yes or no.", user)
    if task_type in {"nli2", "nli3"}:
        labels = ", ".join(LABELS_BY_TASK_TYPE[task_type])
        user = f"""Classify the relationship between the premise and hypothesis.

Premise:
{row["premise"]}

Hypothesis:
{row["hypothesis"]}

Return exactly one label: {labels}.
Label:"""
        return render_chat(tokenizer, f"You are a textual entailment classifier. Return only {labels}.", user)
    raise ValueError(f"unsupported task_type: {task_type}")


def parse_label(text: str, labels: list[str]) -> str | None:
    normalized = text.strip().lower().replace("-", "_")
    normalized_space = normalized.replace("_", " ")
    for label in sorted(labels, key=len, reverse=True):
        label_norm = label.lower().replace("-", "_")
        variants = {label_norm, label_norm.replace("_", " ")}
        if any(normalized.startswith(variant) or normalized_space.startswith(variant) for variant in variants):
            return label
    if labels == ["A", "B", "C", "D"]:
        first = text.strip().upper()[:1]
        return first if first in labels else None
    return None


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def entropy_norm(preds: list[str], labels: list[str]) -> float:
    counts = Counter(preds)
    total = len(preds)
    if total <= 0 or len(labels) <= 1:
        return 0.0
    entropy = 0.0
    for label in labels + ["parse_failure"]:
        count = counts[label]
        if count:
            prob = count / total
            entropy -= prob * math.log(prob)
    return entropy / math.log(len(labels) + (1 if counts["parse_failure"] else 0))


def per_class_metrics(gold: list[str], pred: list[str], labels: list[str]) -> list[dict]:
    rows = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "label": label,
                "support": sum(g == label for g in gold),
                "pred_count": sum(p == label for p in pred),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--template-id", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate schema and labels without loading a model.")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.data_file))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("no rows to evaluate")

    task_types = {row["task_type"] for row in rows}
    if len(task_types) != 1:
        raise ValueError(f"run one task_type per file; found {sorted(task_types)}")
    task_type = next(iter(task_types))
    labels = rows[0].get("labels") or LABELS_BY_TASK_TYPE[task_type]

    if args.dry_run:
        gold_counts = Counter(row["gold"] for row in rows)
        bad_labels = sorted(label for label in gold_counts if label not in labels)
        summary = {
            "data_file": args.data_file,
            "task_type": task_type,
            "n": len(rows),
            "labels": labels,
            "gold_counts": dict(gold_counts),
            "bad_gold_labels": bad_labels,
            "status": "ok" if not bad_labels else "bad_labels",
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if not bad_labels else 1

    import torch
    from peft import PeftModel
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    predictions: list[dict] = []
    for row in tqdm(rows, desc=f"external eval {Path(args.data_file).stem}"):
        prompt = prompt_for_row(tokenizer, row, template_id=args.template_id)
        raw = generate_text(model, tokenizer, prompt, args.max_new_tokens)
        pred = parse_label(raw, labels) or "parse_failure"
        predictions.append(
            {
                "id": row["id"],
                "benchmark": row.get("benchmark", ""),
                "task_type": task_type,
                "gold": row["gold"],
                "pred": pred,
                "raw": raw,
                "source_dataset": row.get("source_dataset", ""),
                "source_config": row.get("source_config", ""),
                "source_split": row.get("source_split", ""),
                "subject": row.get("subject", ""),
            }
        )

    out_path = Path(args.out)
    write_jsonl(predictions, out_path)

    gold = [row["gold"] for row in predictions]
    pred = [row["pred"] for row in predictions]
    pred_counts = Counter(pred)
    gold_counts = Counter(gold)
    dominant_label, dominant_count = max(pred_counts.items(), key=lambda item: item[1])
    parse_failures = pred_counts["parse_failure"]
    valid_pairs = [(g, p) for g, p in zip(gold, pred) if p != "parse_failure"]
    valid_gold = [g for g, _ in valid_pairs]
    valid_pred = [p for _, p in valid_pairs]

    metrics = {
        "data_file": args.data_file,
        "model": args.model,
        "adapter": args.adapter or "",
        "task_type": task_type,
        "n": len(predictions),
        "labels": labels,
        "accuracy_all_outputs": accuracy_score(gold, pred),
        "macro_f1_all_outputs": f1_score(gold, pred, labels=labels, average="macro", zero_division=0),
        "accuracy_parsed_outputs": accuracy_score(valid_gold, valid_pred) if valid_pairs else 0.0,
        "macro_f1_parsed_outputs": f1_score(valid_gold, valid_pred, labels=labels, average="macro", zero_division=0)
        if valid_pairs
        else 0.0,
        "parse_failures": parse_failures,
        "gold_counts": dict(gold_counts),
        "pred_counts": dict(pred_counts),
        "dominant_label": dominant_label,
        "dominant_rate": dominant_count / len(predictions),
        "entropy": entropy_norm(pred, labels),
        "collapse_flag": (dominant_count / len(predictions)) >= 0.80
        or sum(pred_counts[label] == 0 for label in labels) >= max(1, len(labels) // 2),
    }
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    per_class_path = out_path.with_suffix(".per_class.csv")
    write_csv(per_class_metrics(gold, pred, labels), per_class_path)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"wrote predictions: {out_path}")
    print(f"wrote metrics: {metrics_path}")
    print(f"wrote per-class metrics: {per_class_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
