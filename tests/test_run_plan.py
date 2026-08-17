from it_eval_framework.config import EvaluationConfig
from it_eval_framework.runners.plan import print_evaluation_plan


def test_plan_explains_disabled_lighteval_and_unbounded_blimp(capsys):
    config = EvaluationConfig.model_validate(
        {
            "model": {"source": "owner/model"},
            "lighteval": {"enabled": False, "suite": None, "max_samples": None},
            "blimp_it": {"enabled": True, "max_samples": None},
            "perplexity": None,
            "generation": {"enabled": False},
        }
    )

    print_evaluation_plan(config)
    output = capsys.readouterr().out

    assert "LightEval: disabled" in output
    assert "subset=all dataset subsets" in output
    assert "all available" in output


def test_plan_lists_every_selected_lighteval_task(capsys):
    config = EvaluationConfig.model_validate(
        {
            "model": {"source": "owner/model"},
            "lighteval": {"enabled": True, "suite": "quick", "max_samples": 2},
            "blimp_it": {"enabled": False},
            "perplexity": None,
            "generation": {"enabled": False},
        }
    )

    print_evaluation_plan(config)
    output = capsys.readouterr().out

    assert "suite=quick; tasks=1; at most 2 samples per task" in output
    assert "squad_it.default -> squad_ita" in output
