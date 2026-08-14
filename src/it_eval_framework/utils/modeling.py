from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from it_eval_framework.config import ModelConfig


DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_tokenizer(model_config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_id,
        revision=model_config.tokenizer_revision,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=model_config.tokenizer_use_fast,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_config: ModelConfig):
    torch_dtype = DTYPE_MAP[model_config.dtype]
    kwargs = {
        "revision": model_config.revision,
        "trust_remote_code": model_config.trust_remote_code,
    }
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_config.source, **kwargs)
    model.eval()
    if model_config.device not in {"auto", "cpu"}:
        model.to(model_config.device)
    return model


def summarize_model_source(source: str) -> tuple[str, str]:
    path = Path(source)
    if path.exists():
        model_name = path.name or "local_model"
        checkpoint_name = path.name or "checkpoint"
    else:
        model_name = source.replace("/", "__")
        checkpoint_name = source.split("/")[-1]
    return model_name, checkpoint_name
