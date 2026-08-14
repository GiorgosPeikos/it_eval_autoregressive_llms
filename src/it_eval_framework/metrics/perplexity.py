from __future__ import annotations

from dataclasses import dataclass

import math
import torch


@dataclass
class WindowStat:
    negative_log_likelihood: float
    token_count: int


def sliding_windows(token_count: int, sequence_length: int, stride: int):
    start = 0
    while start < token_count:
        end = min(start + sequence_length, token_count)
        yield start, end
        if end == token_count:
            break
        start += stride


def compute_window_nll(
    model,
    input_ids: torch.Tensor,
    start: int,
    end: int,
    device: str = "cpu",
    target_start: int | None = None,
) -> WindowStat:
    chunk = input_ids[:, start:end].to(device)
    labels = chunk.clone()
    if target_start is not None:
        context_length = max(target_start - start, 0)
        labels[:, :context_length] = -100
    with torch.no_grad():
        outputs = model(input_ids=chunk, labels=labels)
    effective_tokens = int((labels[:, 1:] != -100).sum().item())
    nll = float(outputs.loss.item() * effective_tokens)
    return WindowStat(negative_log_likelihood=nll, token_count=effective_tokens)


def finalize_perplexity(total_nll: float, total_tokens: int) -> dict:
    if total_tokens <= 0:
        raise ValueError("Perplexity requires at least one scored target token.")
    mean_loss = total_nll / total_tokens
    return {
        "total_negative_log_likelihood": total_nll,
        "total_token_count": total_tokens,
        "mean_loss": mean_loss,
        "token_perplexity": math.exp(mean_loss),
    }
