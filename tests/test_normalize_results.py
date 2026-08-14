from it_eval_framework.reporting.normalize_results import normalize_lighteval_results


def test_normalize_lighteval_results_pairs_values_and_stderr():
    raw = {
        "results": {
            "xcopa_ita_cf|0": {
                "acc_norm": 0.5,
                "acc_norm_stderr": 0.1,
                "label": "ignored",
            }
        }
    }

    assert normalize_lighteval_results(raw) == [
        {
            "component": "lighteval",
            "task_id": "xcopa_ita_cf",
            "fewshot": 0,
            "metric": "acc_norm",
            "value": 0.5,
            "stderr": 0.1,
        }
    ]
