from pathlib import Path

from it_eval_framework import api
from it_eval_framework.config import load_config


def test_evaluate_accepts_config_path(monkeypatch):
    monkeypatch.setattr(api, "run_all", lambda path: Path(path))

    assert api.evaluate("configs/italian_base_quick.yaml") == Path("configs/italian_base_quick.yaml")


def test_evaluate_accepts_config_object(monkeypatch):
    observed = {}

    def fake_run(path):
        observed["payload"] = Path(path).read_text(encoding="utf-8")
        return Path("results")

    monkeypatch.setattr(api, "run_all", fake_run)
    result = api.evaluate(load_config("configs/italian_base_quick.yaml"))

    assert result == Path("results")
    assert "italian_base_quick" in observed["payload"]
