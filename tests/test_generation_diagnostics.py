from it_eval_framework.metrics.generation_diagnostics import summarize_generation


def test_generation_diagnostics_flags_unfinished():
    payload = summarize_generation("Questo testo non finisce")
    assert payload["unfinished_output"] is True
