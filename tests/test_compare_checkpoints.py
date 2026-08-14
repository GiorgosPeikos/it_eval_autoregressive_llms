import pandas as pd
import pytest

from it_eval_framework.reporting.compare_checkpoints import compare_summaries, comparison_warnings


def _summary(values):
    return pd.DataFrame(
        [
            {"component": "lighteval", "task_id": task, "fewshot": 0, "metric": "acc", "value": value}
            for task, value in values
        ]
    )


def test_comparison_matches_normalized_metrics_by_task():
    compared = compare_summaries(_summary([("a", 0.1), ("b", 0.2)]), _summary([("a", 0.3), ("b", 0.4)]))

    assert len(compared) == 2
    assert dict(zip(compared.task_id, compared.difference)) == pytest.approx({"a": 0.2, "b": 0.2})


def test_comparison_rejects_duplicate_result_identities():
    duplicated = _summary([("a", 0.1), ("a", 0.2)])

    with pytest.raises(ValueError, match="duplicate result identities"):
        compare_summaries(duplicated, _summary([("a", 0.3)]))


def test_comparison_warns_when_evaluation_settings_differ():
    left = {"model": {"source": "model/a", "tokenizer_source": "tokenizer/a"}, "perplexity": {"enabled": True, "split": "validation"}}
    right = {"model": {"source": "model/b", "tokenizer_source": "tokenizer/b"}, "perplexity": {"enabled": True, "split": "test"}}

    messages = comparison_warnings(left, right)

    assert "Evaluation setting differs: perplexity.split" in messages
    assert "Evaluation setting differs: effective tokenizer" in messages
