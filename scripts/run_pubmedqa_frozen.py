from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from medself.data import flatten_pubmedqa_context, load_pubmedqa
from medself.inference import predict_label
from medself.prompts import build_pubmedqa_prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template-id", type=int, default=0)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default="runs/pubmedqa_frozen_smoke.jsonl")
    args = parser.parse_args()

    datasets = load_pubmedqa(seed=args.seed)
    eval_set = datasets[args.split].select(range(min(args.max_samples, len(datasets[args.split]))))

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

    gold_labels: list[str] = []
    pred_labels: list[str] = []
    parse_failures = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(eval_set, desc="pubmedqa frozen"):
            context = flatten_pubmedqa_context(example)
            prompt = build_pubmedqa_prompt(tokenizer, example["question"], context, args.template_id)
            pred, raw = predict_label(model, tokenizer, prompt, args.max_new_tokens)
            if pred is None:
                parse_failures += 1
                pred = "parse_failure"

            gold = example["final_decision"]
            gold_labels.append(gold)
            pred_labels.append(pred)

            handle.write(
                json.dumps(
                    {
                        "pubid": example["pubid"],
                        "question": example["question"],
                        "gold": gold,
                        "pred": pred,
                        "raw": raw,
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
    accuracy = accuracy_score(
        [gold for gold, _ in valid_pairs], [pred for _, pred in valid_pairs]
    ) if valid_pairs else 0.0

    print(f"samples: {len(eval_set)}")
    print(f"parse failures: {parse_failures}")
    print(f"accuracy on parsed outputs: {accuracy:.4f}")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
