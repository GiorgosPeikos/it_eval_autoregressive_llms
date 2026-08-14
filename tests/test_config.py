from pathlib import Path

from it_eval_framework.config import load_config


def test_quick_config_loads():
    config = load_config(Path("configs/italian_base_quick.yaml"))
    assert config.model.source == "sshleifer/tiny-gpt2"
    assert config.lighteval.enabled is False
    assert config.lighteval.task_aliases == ["squad_it.default"]
    assert config.perplexity.dataset_repo == "gsarti/clean_mc4_it"
    assert config.perplexity.dataset_subset == "tiny"
    assert config.model.revision == config.model.tokenizer_revision
    assert len(config.perplexity.dataset_revision) == 40


def test_perplexity_rejects_ambiguous_dataset_source(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  source: test/model
perplexity:
  dataset_path: corpus.txt
  dataset_repo: owner/corpus
""",
        encoding="utf-8",
    )

    try:
        load_config(config_path)
    except ValueError as error:
        assert "Set only one" in str(error)
    else:
        raise AssertionError("Expected ambiguous perplexity dataset source to fail")
