from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SequenceScore:
    total_logprob: float
    token_count: int
    char_count: int

    @property
    def avg_logprob_per_token(self) -> float:
        return self.total_logprob / max(self.token_count, 1)

    @property
    def avg_logprob_per_char(self) -> float:
        return self.total_logprob / max(self.char_count, 1)


def score_full_sequence(model, tokenizer, text: str, device: str = "cpu") -> SequenceScore:
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    total = token_log_probs.sum().item()
    return SequenceScore(total_logprob=total, token_count=labels.numel(), char_count=len(text))


def choose_preferred(score_a: SequenceScore, score_b: SequenceScore, normalization: str = "raw") -> int:
    if normalization == "token":
        lhs, rhs = score_a.avg_logprob_per_token, score_b.avg_logprob_per_token
    elif normalization == "char":
        lhs, rhs = score_a.avg_logprob_per_char, score_b.avg_logprob_per_char
    else:
        lhs, rhs = score_a.total_logprob, score_b.total_logprob
    return 0 if lhs >= rhs else 1
