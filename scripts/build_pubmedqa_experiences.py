from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from medself.data import flatten_pubmedqa_context, load_pubmedqa
from medself.inference import predict_label
from medself.prompts import build_pubmedqa_prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/pubmedqa_experiences_train_200.jsonl")
    args = parser.parse_args()

    datasets = load_pubmedqa(seed=args.seed)
    split = datasets[args.split]
    eval_set = split.select(range(min(args.max_samples, len(split))))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept_correct = 0
    parsed = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(eval_set, desc="build experiences"):
            context = flatten_pubmedqa_context(example)
            prompt = build_pubmedqa_prompt(tokenizer, example["question"], context)
            pred, raw = predict_label(model, tokenizer, prompt, args.max_new_tokens)
            gold = example["final_decision"]
            is_correct = pred == gold
            parsed += pred is not None
            kept_correct += is_correct
            handle.write(
                json.dumps(
                    {
                        "pubid": example["pubid"],
                        "question": example["question"],
                        "context": context,
                        "gold": gold,
                        "pred": pred,
                        "raw": raw,
                        "verified_correct": is_correct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"samples: {len(eval_set)}")
    print(f"parsed: {parsed}")
    print(f"verified correct: {kept_correct}")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
