from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from medself.data import flatten_pubmedqa_context, load_pubmedqa
from medself.inference import predict_label
from medself.memory import TfidfMemory, load_experience_memory
from medself.prompts import build_pubmedqa_user_text


def truncate(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def build_retrieval_prompt(
    tokenizer,
    question: str,
    context: str,
    exemplars: list[dict],
    exemplar_context_chars: int,
) -> str:
    memory_blocks = []
    for index, item in enumerate(exemplars, start=1):
        memory_blocks.append(
            "\n".join(
                [
                    f"Example {index}:",
                    f"Question: {truncate(item['question'], 280)}",
                    f"Context snippet: {truncate(item['context'], exemplar_context_chars)}",
                    f"Verified answer: {item['label']}",
                ]
            )
        )

    memory_text = "\n\n".join(memory_blocks)
    target_prompt = build_pubmedqa_user_text(question, context)

    user_prompt = f"""You may use the previous verified experiences below as memory. They are not the target question.

{memory_text}

Now solve the target question. Return exactly one label: yes, no, or maybe.

{target_prompt}"""

    messages = [
        {
            "role": "system",
            "content": "You are a biomedical QA classifier with episodic memory. Return only yes, no, or maybe.",
        },
        {"role": "user", "content": user_prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--memory-file", required=True)
    parser.add_argument(
        "--memory-mode",
        default="verifier_all",
        choices=["verifier_all", "self_correct", "self_predicted"],
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--exemplar-context-chars", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/pubmedqa_retrieval_test100.jsonl")
    args = parser.parse_args()

    memory_examples = load_experience_memory(args.memory_file, args.memory_mode)
    if not memory_examples:
        raise ValueError("memory is empty")
    memory = TfidfMemory(memory_examples)

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

    gold_labels: list[str] = []
    pred_labels: list[str] = []
    parse_failures = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(eval_set, desc="pubmedqa retrieval"):
            context = flatten_pubmedqa_context(example)
            query = {"question": example["question"], "context": context}
            exemplars = memory.retrieve(query, args.k)
            prompt = build_retrieval_prompt(
                tokenizer,
                example["question"],
                context,
                exemplars,
                args.exemplar_context_chars,
            )
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
                        "memory_mode": args.memory_mode,
                        "retrieved": [
                            {
                                "question": item["question"],
                                "label": item["label"],
                                "score": item["retrieval_score"],
                            }
                            for item in exemplars
                        ],
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
    accuracy = (
        accuracy_score([gold for gold, _ in valid_pairs], [pred for _, pred in valid_pairs])
        if valid_pairs
        else 0.0
    )
    print(f"memory examples: {len(memory_examples)}")
    print(f"samples: {len(eval_set)}")
    print(f"parse failures: {parse_failures}")
    print(f"accuracy on parsed outputs: {accuracy:.4f}")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
