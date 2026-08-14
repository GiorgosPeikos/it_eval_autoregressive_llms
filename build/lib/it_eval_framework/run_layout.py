from __future__ import annotations

from pathlib import Path

from it_eval_framework.config import EvaluationConfig
from it_eval_framework.utils.hashing import stable_hash
from it_eval_framework.utils.io import ensure_dir
from it_eval_framework.utils.modeling import summarize_model_source


def build_run_directory(config: EvaluationConfig) -> Path:
    model_name, checkpoint_name = summarize_model_source(config.model.source)
    config_hash = stable_hash(config.model_dump(mode="json"))
    run_dir = (
        Path(config.output.root_dir)
        / model_name
        / checkpoint_name
        / f"{config.run_name}_{config_hash}"
    )
    return ensure_dir(run_dir)
