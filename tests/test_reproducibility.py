from pathlib import Path

from it_eval_framework.config import load_config
from it_eval_framework.runners.common import reproducibility_issues


def test_quick_config_has_fully_pinned_configurable_inputs():
    config = load_config(Path("configs/italian_base_quick.yaml"))

    assert reproducibility_issues(config) == []


def test_local_model_requires_user_supplied_artifact_digest():
    config = load_config(Path("configs/local_model_example.yaml"))

    assert "Local model source has no model.artifact_sha256 content digest." in reproducibility_issues(config)
