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


def test_evaluate_builds_bounded_quick_preset(monkeypatch):
    observed = {}

    def fake_run(path):
        observed["config"] = load_config(path)
        return Path("results")

    monkeypatch.setattr(api, "run_all", fake_run)
    api.evaluate(model="owner/model", preset="quick", device="cpu")

    assert observed["config"].model.source == "owner/model"
    assert observed["config"].lighteval.enabled is False
    assert observed["config"].perplexity.max_documents == 3


def test_evaluate_accepts_perplexity_budget_overrides(monkeypatch):
    observed = {}

    def fake_run(path):
        observed["config"] = load_config(path)
        return Path("results")

    monkeypatch.setattr(api, "run_all", fake_run)
    api.evaluate(
        model="owner/model",
        preset="quick",
        device="cpu",
        perplexity_subset="medium",
        perplexity_max_documents=250,
        perplexity_max_tokens_per_document=1024,
    )

    perplexity = observed["config"].perplexity
    assert perplexity.dataset_subset == "medium"
    assert perplexity.max_documents == 250
    assert perplexity.max_tokens_per_document == 1024
