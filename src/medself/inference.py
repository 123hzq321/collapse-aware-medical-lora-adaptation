from __future__ import annotations

import torch

from medself.prompts import parse_pubmedqa_label


def predict_label(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 8,
) -> tuple[str | None, str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_pubmedqa_label(text), text


def parse_choice_label(text: str) -> str | None:
    normalized = text.strip().upper()
    for token in ("A", "B", "C", "D"):
        if normalized.startswith(token):
            return token
    return None


def predict_choice(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 8,
) -> tuple[str | None, str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return parse_choice_label(text), text
