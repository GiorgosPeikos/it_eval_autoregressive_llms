from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath

from huggingface_hub import HfApi

from it_eval_framework.config import EvaluationConfig


COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def is_local_source(source: str) -> bool:
    return (
        Path(source).exists()
        or bool(PureWindowsPath(source).drive)
        or source.startswith(("./", "../", "/"))
    )


def resolve_repo_revision(repo_id: str, revision: str | None, api: HfApi | None = None) -> str:
    """Resolve a Hub branch, tag, or omitted revision to an immutable commit SHA."""
    if revision and COMMIT_SHA_PATTERN.fullmatch(revision):
        return revision.lower()

    requested_revision = revision or "main"
    info = (api or HfApi()).model_info(repo_id, revision=requested_revision)
    if not info.sha:
        raise ValueError(f"Hugging Face did not return a commit SHA for {repo_id!r} at {requested_revision!r}.")
    return info.sha


def resolve_model_revisions(config: EvaluationConfig, api: HfApi | None = None) -> EvaluationConfig:
    """Return a copy with remote model and tokenizer revisions pinned independently."""
    resolved = config.model_copy(deep=True)
    model = resolved.model

    if not is_local_source(model.source):
        model.revision = resolve_repo_revision(model.source, model.revision, api)

    tokenizer_source = model.tokenizer_id
    if not is_local_source(tokenizer_source):
        if tokenizer_source == model.source and model.tokenizer_revision is None:
            model.tokenizer_revision = model.revision
        else:
            model.tokenizer_revision = resolve_repo_revision(
                tokenizer_source,
                model.tokenizer_revision,
                api,
            )

    return resolved
