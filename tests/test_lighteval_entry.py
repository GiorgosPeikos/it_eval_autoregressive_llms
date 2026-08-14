from pathlib import Path

from lighteval.models.model_output import ModelResponse

from it_eval_framework.runners.lighteval_entry import (
    _merge_unconditioned_response,
    _sanitize_filename_component,
    _sanitize_windows_cache_path,
    _to_windows_extended_path,
)


def test_sanitize_filename_component_replaces_windows_invalid_chars():
    assert _sanitize_filename_component('xcopa_ita_cf|0:"test"?*') == "xcopa_ita_cf__0____test______"


def test_to_windows_extended_path_returns_absolute_path():
    value = _to_windows_extended_path(Path("configs/italian_base_quick.yaml"))
    assert value.endswith("configs\\italian_base_quick.yaml")


def test_sanitize_windows_cache_path_preserves_drive_anchor():
    value = _sanitize_windows_cache_path(r"C:\Users\User\.cache\huggingface\lighteval\xcopa_ita_cf|0")
    assert str(value) == r"C:\Users\User\.cache\huggingface\lighteval\xcopa_ita_cf__0"


def test_merge_unconditioned_response_appends_logprobs():
    conditioned = ModelResponse(logprobs=[-1.0, -2.0], output_tokens=[[1], [2]])
    unconditioned = ModelResponse(logprobs=[-0.5, -1.5], output_tokens=[[1], [2]])

    merged = _merge_unconditioned_response(conditioned, unconditioned)

    assert merged.logprobs == [-1.0, -2.0, -0.5, -1.5]
    assert merged.unconditioned_logprobs == [-0.5, -1.5]
