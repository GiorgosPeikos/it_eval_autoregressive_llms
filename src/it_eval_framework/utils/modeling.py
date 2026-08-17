from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer

from it_eval_framework.config import ModelConfig


DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_tokenizer_file(model_config: ModelConfig) -> str | None:
    """Return an absolute tokenizer.json path when the tokenizer provides one.

    Transformers 4.57.x can let a relative ``tokenizer_file`` value from
    tokenizer_config.json override the absolute Hub cache path. Supplying the
    resolved path explicitly avoids that upstream path-resolution bug.
    """
    if not model_config.tokenizer_use_fast:
        return None

    tokenizer_id = model_config.tokenizer_id
    local_source = Path(tokenizer_id)
    if local_source.exists():
        candidate = local_source / "tokenizer.json" if local_source.is_dir() else local_source
        return str(candidate.resolve()) if candidate.is_file() else None

    try:
        return hf_hub_download(
            repo_id=tokenizer_id,
            filename="tokenizer.json",
            revision=model_config.tokenizer_revision,
        )
    except EntryNotFoundError:
        # Slow-tokenizer repositories may legitimately omit tokenizer.json.
        return None


def load_tokenizer(model_config: ModelConfig):
    kwargs = {
        "revision": model_config.tokenizer_revision,
        "trust_remote_code": model_config.trust_remote_code,
        "use_fast": model_config.tokenizer_use_fast,
    }
    tokenizer_file = resolve_tokenizer_file(model_config)
    if tokenizer_file:
        kwargs["tokenizer_file"] = tokenizer_file
    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_id, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_generation_inputs(tokenizer, prompt: str, device: str):
    """Tokenize a single decoder-only prompt with generation-safe inputs."""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    # Some custom tokenizers ignore return_token_type_ids. Segment IDs are not
    # needed for a single decoder-only prompt and many causal LMs reject them.
    encoded.pop("token_type_ids", None)
    return encoded.to(device)


def load_model(model_config: ModelConfig):
    torch_dtype = DTYPE_MAP[model_config.dtype]
    kwargs = {
        "revision": model_config.revision,
        "trust_remote_code": model_config.trust_remote_code,
    }
    if torch_dtype != "auto":
        kwargs["torch_dtype"] = torch_dtype
    print(
        f"[model] Loading model '{model_config.source}' with dtype={model_config.dtype} on device={model_config.device}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.source, **kwargs)
    model.eval()
    if model_config.device not in {"auto", "cpu"}:
        model.to(model_config.device)
    resolved_device = next(model.parameters()).device
    print(f"[model] Model ready on device={resolved_device}", flush=True)
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
