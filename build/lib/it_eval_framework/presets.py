from __future__ import annotations

from pathlib import Path

import torch

from it_eval_framework.config import EvaluationConfig


PRESET_NAMES = ("quick", "perplexity", "verified_windows", "full")
BLIMP_REVISION = "4159ecb68388283488cb1d235a7e1946489bc62d"
PPL_REVISION = "167d5696e91ac89f17936f9d0059031cbc4c9e99"
PACKAGED_PROMPTS = Path(__file__).parent / "resources" / "generation_prompts.yaml"


def build_evaluation_config(
    model: str,
    *,
    preset: str = "quick",
    tokenizer: str | None = None,
    revision: str | None = None,
    tokenizer_revision: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 1,
    output_dir: str | Path = "evaluation_results",
    run_name: str | None = None,
    artifact_sha256: str | None = None,
) -> EvaluationConfig:
    if preset not in PRESET_NAMES:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {', '.join(PRESET_NAMES)}")
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    is_full = preset == "full"
    is_ppl_only = preset == "perplexity"
    is_lighteval_only = preset == "verified_windows"
    payload = {
        "run_name": run_name or f"{preset}_evaluation",
        "model": {
            "source": model,
            "tokenizer_source": tokenizer or model,
            "revision": revision,
            "tokenizer_revision": tokenizer_revision or revision,
            "artifact_sha256": artifact_sha256,
            "device": resolved_device,
            "dtype": dtype,
            "batch_size": batch_size,
        },
        "output": {"root_dir": str(output_dir), "overwrite": False, "save_details": True},
        "runtime": {"seed": 13},
        "lighteval": {
            "enabled": is_full or is_lighteval_only,
            "suite": "full" if is_full else ("verified_windows" if is_lighteval_only else None),
            "max_samples": None if is_full else 2,
            "dataset_loading_processes": 1,
        },
        "blimp_it": {
            "enabled": is_full or (not is_ppl_only and not is_lighteval_only),
            "dataset_revision": BLIMP_REVISION,
            "max_samples": None if is_full else 20,
        },
        "perplexity": {
            "enabled": is_full or is_ppl_only or preset == "quick",
            "dataset_repo": "gsarti/clean_mc4_it",
            "dataset_subset": "tiny",
            "dataset_revision": PPL_REVISION,
            "dataset_trust_remote_code": True,
            "dataset_streaming": not is_full,
            "split": "validation",
            "text_field": "text",
            "sequence_length": 1024 if is_full else 512,
            "stride": 512 if is_full else 256,
            "max_documents": None if is_full else 3,
            "max_tokens_per_document": None if is_full else 256,
        },
        "generation": {
            "enabled": is_full or (not is_ppl_only and not is_lighteval_only),
            "prompts_path": str(PACKAGED_PROMPTS),
            "max_prompts": None if is_full else 3,
            "seed": 13,
        },
    }
    return EvaluationConfig.model_validate(payload)
