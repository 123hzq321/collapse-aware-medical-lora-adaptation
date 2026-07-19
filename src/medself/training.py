from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass
class PubMedQASFTItem:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    sample_weight: float = 1.0


class PubMedQASFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length: int = 1536):
        from medself.data import flatten_pubmedqa_context
        from medself.prompts import build_pubmedqa_prompt

        self.items: list[PubMedQASFTItem] = []
        eos = tokenizer.eos_token or ""

        for example in examples:
            context = flatten_pubmedqa_context(example)
            prompt = build_pubmedqa_prompt(tokenizer, example["question"], context)
            target = f"{example['final_decision']}{eos}"

            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            budget = max_length - len(target_ids)
            if budget <= 0:
                continue

            if len(prompt_ids) > budget:
                prompt_ids = prompt_ids[-budget:]

            input_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
            labels = torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            self.items.append(PubMedQASFTItem(input_ids, attention_mask, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> PubMedQASFTItem:
        return self.items[index]


class TextSFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length: int = 1536):
        self.items: list[PubMedQASFTItem] = []
        eos = tokenizer.eos_token or ""

        for example in examples:
            prompt = example["prompt"]
            target = f"{example['target']}{eos}"

            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            budget = max_length - len(target_ids)
            if budget <= 0:
                continue

            if len(prompt_ids) > budget:
                prompt_ids = prompt_ids[-budget:]

            input_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
            labels = torch.tensor([-100] * len(prompt_ids) + target_ids, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            sample_weight = float(example.get("sample_weight", 1.0))
            self.items.append(PubMedQASFTItem(input_ids, attention_mask, labels, sample_weight))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> PubMedQASFTItem:
        return self.items[index]


def collate_sft_batch(
    batch: list[PubMedQASFTItem],
    pad_token_id: int,
    include_sample_weights: bool = False,
) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence(
        [item.input_ids for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [item.attention_mask for item in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [item.labels for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if include_sample_weights:
        collated["sample_weights"] = torch.tensor(
            [item.sample_weight for item in batch],
            dtype=torch.float32,
        )
    return collated


def compute_weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.shape)
    label_mask = shift_labels.ne(-100)
    per_sample_loss = (token_losses * label_mask).sum(dim=1) / label_mask.sum(dim=1).clamp_min(1)
    return (per_sample_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
