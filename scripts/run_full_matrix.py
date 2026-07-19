from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


TRAIN_FILE = "data/multitask/sft_pubmedqa_x6_medmcqa30k_medqa10k_seed42.jsonl"
SEEDS = [42, 43, 44]


@dataclass(frozen=True)
class ModelSpec:
    short: str
    hf_id: str
    max_length: int
    grad_accum: int
    gradient_checkpointing: bool


MODEL_SPECS = {
    "qwen05b": ModelSpec(
        short="qwen05b",
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        max_length=1536,
        grad_accum=8,
        gradient_checkpointing=False,
    ),
    "qwen15b": ModelSpec(
        short="qwen15b",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        max_length=1024,
        grad_accum=8,
        gradient_checkpointing=True,
    ),
    "qwen3b": ModelSpec(
        short="qwen3b",
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        max_length=768,
        grad_accum=16,
        gradient_checkpointing=True,
    ),
}


def run_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n### START {started}\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n### END {datetime.now(timezone.utc).isoformat()} code={proc.returncode}\n")
    return proc.returncode


def train_command(
    python: str,
    spec: ModelSpec,
    seed: int,
    adapter_dir: str,
    train_file: str,
    max_train_samples: int | None,
    use_sample_weights: bool,
    sample_weight_field: str,
) -> list[str]:
    command = [
        python,
        "scripts/train_text_lora.py",
        "--model",
        spec.hf_id,
        "--train-file",
        train_file,
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--grad-accum",
        str(spec.grad_accum),
        "--max-length",
        str(spec.max_length),
        "--seed",
        str(seed),
        "--out-dir",
        adapter_dir,
    ]
    if max_train_samples is not None:
        command.extend(["--max-train-samples", str(max_train_samples)])
    if spec.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if use_sample_weights:
        command.extend(["--use-sample-weights", "--sample-weight-field", sample_weight_field])
    return command


def eval_commands(python: str, spec: ModelSpec, seed: int, adapter_dir: str, run_prefix: str) -> list[list[str]]:
    return [
        [
            python,
            "scripts/run_pubmedqa_frozen.py",
            "--model",
            spec.hf_id,
            "--split",
            "test",
            "--max-samples",
            "100",
            "--seed",
            "42",
            "--adapter",
            adapter_dir,
            "--out",
            f"runs/{run_prefix}_pubmedqa_test100.jsonl",
        ],
        [
            python,
            "scripts/run_medmcqa_eval.py",
            "--model",
            spec.hf_id,
            "--data-file",
            "data/medmcqa/validation_1000_seed42.jsonl",
            "--max-samples",
            "1000",
            "--adapter",
            adapter_dir,
            "--out",
            f"runs/{run_prefix}_medmcqa_val1000.jsonl",
        ],
        [
            python,
            "scripts/run_medmcqa_eval.py",
            "--model",
            spec.hf_id,
            "--data-file",
            "data/medqa/test_full_seed42.jsonl",
            "--max-samples",
            "1273",
            "--adapter",
            adapter_dir,
            "--out",
            f"runs/{run_prefix}_medqa_test1273.jsonl",
        ],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen05b", "qwen15b", "qwen3b"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--train-file", default=TRAIN_FILE)
    parser.add_argument("--strategy-tag", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--use-sample-weights", action="store_true")
    parser.add_argument("--sample-weight-field", default="sample_weight")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd()
    python = str(cwd / ".venv" / "Scripts" / "python.exe")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(cwd / "src")

    plan = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        for seed in args.seeds:
            if args.strategy_tag:
                sample_tag = args.strategy_tag
            else:
                sample_tag = "full45k" if args.max_train_samples is None else f"sft{args.max_train_samples}"
            run_prefix = f"{spec.short}_{sample_tag}_seed{seed}"
            adapter_dir = f"adapters/{run_prefix}"
            metadata_path = cwd / adapter_dir / "run_metadata.json"
            train_log = cwd / "runs" / "logs" / f"{run_prefix}_train.log"
            eval_log = cwd / "runs" / "logs" / f"{run_prefix}_eval.log"
            plan.append(
                {
                    "model": model_key,
                    "seed": seed,
                    "run_prefix": run_prefix,
                    "adapter_dir": adapter_dir,
                    "train_file": args.train_file,
                    "strategy_tag": sample_tag,
                    "use_sample_weights": args.use_sample_weights,
                    "sample_weight_field": args.sample_weight_field,
                    "train_log": str(train_log),
                    "eval_log": str(eval_log),
                }
            )

            if not args.eval_only:
                command = train_command(
                    python,
                    spec,
                    seed,
                    adapter_dir,
                    args.train_file,
                    args.max_train_samples,
                    args.use_sample_weights,
                    args.sample_weight_field,
                )
                if args.dry_run:
                    print("TRAIN", " ".join(command))
                elif args.skip_existing and metadata_path.exists():
                    print(f"skip existing train: {adapter_dir}")
                else:
                    code = run_command(command, cwd, env, train_log)
                    if code != 0:
                        print(f"train failed for {run_prefix}; see {train_log}", file=sys.stderr)
                        return code

            if not args.train_only:
                for command in eval_commands(python, spec, seed, adapter_dir, run_prefix):
                    out_index = command.index("--out") + 1
                    out_path = cwd / command[out_index]
                    if args.dry_run:
                        print("EVAL", " ".join(command))
                    elif args.skip_existing and out_path.exists():
                        print(f"skip existing eval: {out_path}")
                    else:
                        code = run_command(command, cwd, env, eval_log)
                        if code != 0:
                            print(f"eval failed for {run_prefix}; see {eval_log}", file=sys.stderr)
                            return code

    if args.dry_run:
        print(f"dry-run plan entries: {len(plan)}")
        return 0

    plan_name = "full_matrix_plan.json" if not args.strategy_tag else f"{args.strategy_tag}_plan.json"
    plan_path = cwd / "runs" / plan_name
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote plan: {plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
