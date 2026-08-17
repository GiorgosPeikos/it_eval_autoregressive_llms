from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from it_eval_framework.config import EvaluationConfig
from it_eval_framework.presets import build_evaluation_config


def run_all(config_path: str) -> Path:
    # Keep the runner import lazy so ``python -m ...runners.run_all`` does not
    # preload the module through package initialization.
    from it_eval_framework.runners.run_all import run

    return run(config_path)


def evaluate(
    config: str | Path | EvaluationConfig | None = None,
    *,
    model: str | None = None,
    preset: str = "quick",
    tokenizer: str | None = None,
    revision: str | None = None,
    tokenizer_revision: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 1,
    output_dir: str | Path = "evaluation_results",
    artifact_sha256: str | None = None,
    perplexity_subset: str | None = None,
    perplexity_max_documents: int | None = None,
    perplexity_max_tokens_per_document: int | None = None,
) -> Path:
    """Run an evaluation through the supported public Python API.

    Pass a YAML path when configuration is managed by a project, or an
    ``EvaluationConfig`` when configuration is assembled in Python.
    """
    if config is not None and model is not None:
        raise ValueError("Pass either config or model, not both.")
    if config is None:
        if model is None:
            raise ValueError("Pass a configuration or a model source.")
        config = build_evaluation_config(
            model,
            preset=preset,
            tokenizer=tokenizer,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            output_dir=output_dir,
            artifact_sha256=artifact_sha256,
            perplexity_subset=perplexity_subset,
            perplexity_max_documents=perplexity_max_documents,
            perplexity_max_tokens_per_document=perplexity_max_tokens_per_document,
        )
    if isinstance(config, (str, Path)):
        return run_all(str(config))

    with tempfile.TemporaryDirectory(prefix="it_eval_") as temporary_directory:
        config_path = Path(temporary_directory) / "evaluation.yaml"
        config_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return run_all(str(config_path))
