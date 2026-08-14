import sys
from pathlib import Path

from it_eval_framework import cli


def test_cli_evaluate_uses_public_api(monkeypatch, capsys):
    monkeypatch.setattr(cli, "evaluate", lambda config: Path("evaluation_results/run"))
    monkeypatch.setattr(sys, "argv", ["it-eval", "evaluate", "--config", "model.yaml"])

    cli.main()

    assert capsys.readouterr().out.strip() == str(Path("evaluation_results/run"))


def test_cli_accepts_model_and_preset(monkeypatch, capsys):
    observed = {}

    def fake_evaluate(**kwargs):
        observed.update(kwargs)
        return Path("results")

    monkeypatch.setattr(cli, "evaluate", fake_evaluate)
    monkeypatch.setattr(sys, "argv", ["it-eval", "evaluate", "--model", "owner/model", "--preset", "perplexity"])

    cli.main()

    assert observed["model"] == "owner/model"
    assert observed["preset"] == "perplexity"
