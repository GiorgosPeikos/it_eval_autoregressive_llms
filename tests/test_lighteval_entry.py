from pathlib import Path, PureWindowsPath

from lighteval.models.model_output import ModelResponse

from it_eval_framework.runners.lighteval_entry import (
    _merge_unconditioned_response,
    _resolve_tokenizer_json,
    _sanitize_filename_component,
    _sanitize_windows_cache_path,
    _to_windows_extended_path,
)


def test_sanitize_filename_component_replaces_windows_invalid_chars():
    assert _sanitize_filename_component('xcopa_ita_cf|0:"test"?*') == "xcopa_ita_cf__0____test______"


def test_to_windows_extended_path_returns_absolute_path():
    value = _to_windows_extended_path(Path("configs/italian_base_quick.yaml"))
    assert Path(value).is_absolute()
    assert Path(value).parts[-2:] == ("configs", "italian_base_quick.yaml")


def test_sanitize_windows_cache_path_preserves_drive_anchor():
    value = _sanitize_windows_cache_path(r"C:\Users\User\.cache\huggingface\lighteval\xcopa_ita_cf|0")
    assert PureWindowsPath(value) == PureWindowsPath(
        r"C:\Users\User\.cache\huggingface\lighteval\xcopa_ita_cf__0"
    )


def test_merge_unconditioned_response_appends_logprobs():
    conditioned = ModelResponse(logprobs=[-1.0, -2.0], output_tokens=[[1], [2]])
    unconditioned = ModelResponse(logprobs=[-0.5, -1.5], output_tokens=[[1], [2]])

    merged = _merge_unconditioned_response(conditioned, unconditioned)

    assert merged.logprobs == [-1.0, -2.0, -0.5, -1.5]
    assert merged.unconditioned_logprobs == [-0.5, -1.5]


def test_resolve_tokenizer_json_returns_absolute_local_file(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_json.write_text("{}", encoding="utf-8")

    assert _resolve_tokenizer_json(str(tokenizer_dir), None) == str(tokenizer_json.resolve())
