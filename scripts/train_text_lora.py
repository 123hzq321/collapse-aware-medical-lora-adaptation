from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from medself.training import TextSFTDataset, collate_sft_batch, compute_weighted_causal_lm_loss


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-sample-weights", action="store_true")
    parser.add_argument("--sample-weight-field", default="sample_weight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="adapters/multitask_qwen05b_lora")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    examples = read_jsonl(Path(args.train_file))
    raw_input_examples = len(examples)
    if args.max_train_samples is not None:
        examples = examples[: args.max_train_samples]
    if args.use_sample_weights and args.sample_weight_field != "sample_weight":
        for example in examples:
            example["sample_weight"] = float(example.get(args.sample_weight_field, 1.0))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TextSFTDataset(examples, tokenizer, max_length=args.max_length)
    if not train_dataset:
        raise ValueError("no training examples after tokenization")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda batch: collate_sft_batch(
            batch,
            tokenizer.pad_token_id,
            include_sample_weights=args.use_sample_weights,
        ),
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
            sample_weights = batch.pop("sample_weights", None)
            outputs = model(**batch)
            if sample_weights is None:
                step_loss = outputs.loss
            else:
                step_loss = compute_weighted_causal_lm_loss(
                    outputs.logits,
                    batch["labels"],
                    sample_weights,
                )
            loss = step_loss / args.grad_accum
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
        "train_file": args.train_file,
        "seed": args.seed,
        "raw_input_examples": raw_input_examples,
        "raw_examples": len(examples),
        "train_items": len(train_dataset),
        "max_train_samples": args.max_train_samples,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "global_steps": global_step,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "gradient_checkpointing": args.gradient_checkpointing,
        "use_sample_weights": args.use_sample_weights,
        "sample_weight_field": args.sample_weight_field,
        "lr": args.lr,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
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
