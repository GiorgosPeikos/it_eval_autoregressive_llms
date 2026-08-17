from pathlib import Path

from it_eval_framework.config import discover_evaluation_configs, load_config
from it_eval_framework.config import EvaluationConfig
from it_eval_framework.task_registry import (
    ALL_EVALUABLE_ITALIAN_LIGHTEVAL_ALIASES,
    NON_EVALUABLE_LIGHTEVAL_ALIASES,
)


def test_quick_config_loads():
    config = load_config(Path("configs/italian_base_quick.yaml"))
    assert config.model.source == "Gpeik/Sophira-360M-base"
    assert config.lighteval.enabled is False
    assert config.lighteval.task_aliases == ["squad_it.default"]
    assert config.perplexity.dataset_repo == "gsarti/clean_mc4_it"
    assert config.perplexity.dataset_subset == "tiny"
    assert config.perplexity.dataset_streaming is True
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


def test_all_discovered_evaluation_configs_load():
    paths = discover_evaluation_configs("configs")

    assert Path("configs/generation_prompts.yaml") not in paths
    assert Path("configs/italian_base_quick.yaml") in paths
    for path in paths:
        load_config(path)


def test_all_lighteval_suite_expands_every_registered_alias():
    config = EvaluationConfig.model_validate(
        {
            "model": {"source": "owner/model"},
            "lighteval": {"enabled": True, "suite": "all"},
            "blimp_it": {"enabled": False},
            "perplexity": None,
            "generation": {"enabled": False},
        }
    )

    assert config.lighteval.task_aliases == ALL_EVALUABLE_ITALIAN_LIGHTEVAL_ALIASES
    assert not (set(config.lighteval.task_aliases) & NON_EVALUABLE_LIGHTEVAL_ALIASES)


def test_enabled_lighteval_requires_tasks():
    try:
        EvaluationConfig.model_validate(
            {
                "model": {"source": "owner/model"},
                "lighteval": {"enabled": True, "suite": None},
                "blimp_it": {"enabled": False},
                "perplexity": None,
                "generation": {"enabled": False},
            }
        )
    except ValueError as error:
        assert "requires a suite" in str(error)
    else:
        raise AssertionError("Expected enabled task-less LightEval config to fail")
