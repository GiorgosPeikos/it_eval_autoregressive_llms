from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from it_eval_framework.config import EvaluationConfig
from it_eval_framework.runners.run_all import run as run_all


def evaluate(config: str | Path | EvaluationConfig) -> Path:
    """Run an evaluation through the supported public Python API.

    Pass a YAML path when configuration is managed by a project, or an
    ``EvaluationConfig`` when configuration is assembled in Python.
    """
    if isinstance(config, (str, Path)):
        return run_all(str(config))

    with tempfile.TemporaryDirectory(prefix="it_eval_") as temporary_directory:
        config_path = Path(temporary_directory) / "evaluation.yaml"
        config_path.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return run_all(str(config_path))
