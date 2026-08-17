from it_eval_framework.reporting.normalize_results import normalize_lighteval_results
from it_eval_framework.reporting.aggregate_results import aggregate
from it_eval_framework.utils.io import write_json


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
            "sample_count": None,
            "higher_is_better": None,
        }
    ]


def test_normalize_preserves_subject_aggregate_and_nonzero_fewshot():
    raw = {
        "results": {
            "mlmm_mmlu_ita_mcf:anatomy|5": {"acc": 0.25},
            "all": {"acc": 0.5, "acc_stderr": 0.05},
        }
    }

    rows = normalize_lighteval_results(raw)

    assert rows == [
        {
            "component": "lighteval",
            "task_id": "all",
            "fewshot": None,
            "metric": "acc",
            "value": 0.5,
            "stderr": 0.05,
            "sample_count": None,
            "higher_is_better": None,
        },
        {
            "component": "lighteval",
            "task_id": "mlmm_mmlu_ita_mcf:anatomy",
            "fewshot": 5,
            "metric": "acc",
            "value": 0.25,
            "stderr": None,
            "sample_count": None,
            "higher_is_better": None,
        },
    ]


def test_normalize_adds_lighteval_denominator_and_direction():
    raw = {
        "results": {"xcopa_ita_cf|0": {"acc": 0.75}},
        "config_tasks": {
            "xcopa_ita_cf|0": {
                "effective_num_docs": 20,
                "metric": [{"metric_name": "acc", "higher_is_better": True}],
            }
        },
    }

    row = normalize_lighteval_results(raw)[0]

    assert row["sample_count"] == 20
    assert row["higher_is_better"] is True


def test_aggregate_supports_legacy_benchmark_payload(tmp_path):
    write_json(tmp_path / "benchmark_results.json", {"resolved_tasks": ["a", "b"]})

    summary = aggregate(tmp_path)

    assert summary.to_dict("records") == [{"component": "lighteval", "metric": "task_count", "value": 2}]
