import sys
from pathlib import Path

from it_eval_framework import cli


def test_cli_evaluate_uses_public_api(monkeypatch, capsys):
    monkeypatch.setattr(cli, "evaluate", lambda config: Path("evaluation_results/run"))
    monkeypatch.setattr(sys, "argv", ["it-eval", "evaluate", "--config", "model.yaml"])

    cli.main()

    assert capsys.readouterr().out.strip() == str(Path("evaluation_results/run"))
