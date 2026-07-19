from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from medself.inference import predict_choice
from medself.prompts import build_medmcqa_prompt


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-file", default="data/medmcqa/validation_1000_seed42.jsonl")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--out", default="runs/medmcqa_eval.jsonl")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.data_file))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gold_labels = []
    pred_labels = []
    parse_failures = 0
    confusion: dict[str, Counter] = defaultdict(Counter)

    with out_path.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc="medmcqa eval"):
            prompt = build_medmcqa_prompt(tokenizer, row["question"], row["choices"])
            pred, raw = predict_choice(model, tokenizer, prompt, args.max_new_tokens)
            if pred is None:
                parse_failures += 1
                pred = "parse_failure"

            gold = row["gold"]
            gold_labels.append(gold)
            pred_labels.append(pred)
            confusion[gold][pred] += 1

            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "gold": gold,
                        "pred": pred,
                        "raw": raw,
                        "subject_name": row.get("subject_name", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    valid_pairs = [
        (gold, pred)
        for gold, pred in zip(gold_labels, pred_labels)
        if pred != "parse_failure"
    ]
    valid_gold = [gold for gold, _ in valid_pairs]
    valid_pred = [pred for _, pred in valid_pairs]
    accuracy = accuracy_score(valid_gold, valid_pred) if valid_pairs else 0.0
    macro_f1 = (
        f1_score(valid_gold, valid_pred, labels=["A", "B", "C", "D"], average="macro")
        if valid_pairs
        else 0.0
    )

    print(f"samples: {len(rows)}")
    print(f"parse failures: {parse_failures}")
    print(f"accuracy on parsed outputs: {accuracy:.4f}")
    print(f"macro_f1 on parsed outputs: {macro_f1:.4f}")
    print(f"pred counts: {dict(Counter(pred_labels))}")
    print("confusion:")
    for gold in ("A", "B", "C", "D"):
        print(f"  {gold}: {dict(confusion[gold])}")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
