from pathlib import Path

from it_eval_framework.config import load_config


def test_quick_config_loads():
    config = load_config(Path("configs/italian_base_quick.yaml"))
    assert config.model.source == "sshleifer/tiny-gpt2"
    assert "mlmm_hellaswag.cf" in config.lighteval.task_aliases
