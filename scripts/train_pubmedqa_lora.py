from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from medself.data import load_pubmedqa
from medself.training import PubMedQASFTDataset, collate_sft_batch


def balance_examples(examples: list[dict], strategy: str, seed: int) -> list[dict]:
    if strategy == "none":
        return examples

    groups: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        groups[example["final_decision"]].append(example)

    rng = random.Random(seed)
    for label_examples in groups.values():
        rng.shuffle(label_examples)

    if strategy == "downsample":
        target = min(len(group) for group in groups.values() if group)
        balanced = []
        for label in ("yes", "no", "maybe"):
            balanced.extend(groups[label][:target])
        rng.shuffle(balanced)
        return balanced

    if strategy == "oversample":
        target = max(len(group) for group in groups.values() if group)
        balanced = []
        for label in ("yes", "no", "maybe"):
            group = groups[label]
            if not group:
                continue
            balanced.extend(group)
            while len([item for item in balanced if item["final_decision"] == label]) < target:
                needed = target - len([item for item in balanced if item["final_decision"] == label])
                balanced.extend(group[:needed])
        rng.shuffle(balanced)
        return balanced

    raise ValueError(f"unknown balance strategy: {strategy}")


def load_examples(args) -> list[dict]:
    if args.experience_file:
        rows = [
            json.loads(line)
            for line in Path(args.experience_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.mode == "gated_positive":
            rows = [row for row in rows if row.get("verified_correct")]
            for row in rows:
                row["final_decision"] = row["gold"]
        elif args.mode == "naive_self_training":
            for row in rows:
                row["final_decision"] = row.get("pred") or row.get("gold")
            rows = [row for row in rows if row.get("final_decision") in {"yes", "no", "maybe"}]
        elif args.mode == "oracle":
            for row in rows:
                row["final_decision"] = row["gold"]
        else:
            raise ValueError(f"unknown mode: {args.mode}")
        return balance_examples(rows[: args.max_train_samples], args.balance_labels, args.seed)

    dataset = load_pubmedqa(seed=args.seed)["train"]
    examples = [dataset[i] for i in range(min(args.max_train_samples, len(dataset)))]
    return balance_examples(examples, args.balance_labels, args.seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--experience-file", default=None)
    parser.add_argument(
        "--mode",
        default="oracle",
        choices=["oracle", "gated_positive", "naive_self_training"],
    )
    parser.add_argument("--max-train-samples", type=int, default=200)
    parser.add_argument(
        "--balance-labels",
        default="none",
        choices=["none", "downsample", "oversample"],
    )
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="adapters/pubmedqa_qwen05b_lora_smoke")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = load_examples(args)
    train_dataset = PubMedQASFTDataset(examples, tokenizer, max_length=args.max_length)
    if not train_dataset:
        raise ValueError("no training examples after filtering")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_sft_batch(batch, tokenizer.pad_token_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps_per_epoch = max(1, (len(loader) + args.grad_accum - 1) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        progress = tqdm(loader, desc=f"train epoch {epoch + 1}")
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0 or step == len(loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(loss=f"{float(loss.detach().cpu()) * args.grad_accum:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    metadata = {
        "model": args.model,
        "mode": args.mode,
        "experience_file": args.experience_file,
        "raw_examples": len(examples),
        "train_items": len(train_dataset),
        "balance_labels": args.balance_labels,
        "global_steps": global_step,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved adapter: {out_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
