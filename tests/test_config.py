from pathlib import Path

from it_eval_framework.config import load_config


def test_quick_config_loads():
    config = load_config(Path("configs/italian_base_quick.yaml"))
    assert config.model.source == "sshleifer/tiny-gpt2"
    assert config.lighteval.enabled is False
    assert config.lighteval.task_aliases == ["squad_it.default"]
