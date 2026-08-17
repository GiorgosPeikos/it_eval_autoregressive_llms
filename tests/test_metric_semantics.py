import pandas as pd

from it_eval_framework.reporting.metric_semantics import annotate_metric_rows, metric_semantics


def test_accuracy_semantics_explain_unit_interval():
    semantics = metric_semantics("lighteval", "acc_norm", True)

    assert semantics["range"] == "0–1"
    assert semantics["direction"] == "higher is better"
    assert "Fraction" in semantics["meaning"]


def test_perplexity_semantics_say_lower_is_better():
    semantics = metric_semantics("perplexity", "token_perplexity", False)

    assert semantics["direction"] == "lower is better"
    assert "not an accuracy" in semantics["meaning"]


def test_annotate_metric_rows_adds_explanations():
    result = annotate_metric_rows(
        pd.DataFrame([{"component": "blimp_it", "metric": "accuracy", "value": 1.0}])
    )

    assert result.loc[0, "range"] == "0–1"
    assert "minimal pairs" in result.loc[0, "meaning"]
