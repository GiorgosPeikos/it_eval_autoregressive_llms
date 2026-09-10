from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from lighteval.models.model_output import ModelResponse

from it_eval_framework.runners.lighteval_entry import (
    _compatible_tokenizer_from_pretrained,
    _environment_flag_enabled,
    _merge_unconditioned_response,
    _normalize_generation_requests,
    _optional_positive_environment_integer,
    _repair_distributed_choice_slice_source,
    _run_greedy_until_with_oom_backoff,
    _resolve_tokenizer_json,
    _sanitize_filename_component,
    _sanitize_windows_cache_path,
    _to_windows_extended_path,
    _token_data_to_host,
)


def test_compatible_tokenizer_falls_back_for_tokenizers_backend(monkeypatch, tmp_path):
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text("{}", encoding="utf-8")
    expected = object()
    observed = {}

    def incompatible_auto_tokenizer(source, *args, **kwargs):
        raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")

    def compatible_fast_tokenizer(source, *args, **kwargs):
        observed.update(source=source, args=args, kwargs=kwargs)
        return expected

    monkeypatch.setattr(
        "transformers.PreTrainedTokenizerFast.from_pretrained",
        compatible_fast_tokenizer,
    )

    result = _compatible_tokenizer_from_pretrained(
        incompatible_auto_tokenizer,
        str(tmp_path),
        revision=None,
        use_fast=True,
    )

    assert result is expected
    assert observed["source"] == str(tmp_path)
    assert observed["kwargs"]["tokenizer_file"] == str(tokenizer_json.resolve())
    assert "use_fast" not in observed["kwargs"]


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


def test_token_data_to_host_detaches_nested_tensor_like_values():
    class TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [1, 2, 3]

    assert _token_data_to_host((TensorLike(), [4])) == [[1, 2, 3], [4]]


def test_repair_distributed_choice_slice_preserves_both_tensor_axes():
    broken = "value = gathered_continuations[i][:num_choices][:len_choice]"

    repaired = _repair_distributed_choice_slice_source(broken)

    assert repaired == "value = gathered_continuations[i][:num_choices, :len_choice]"


def test_repair_distributed_choice_slice_accepts_already_fixed_source():
    fixed = "value = gathered_continuations[i][:num_choices, :len_choice]"

    assert _repair_distributed_choice_slice_source(fixed) == fixed


def test_repair_distributed_choice_slice_rejects_unknown_source():
    with pytest.raises(RuntimeError, match="expected one distributed choice-slice pattern"):
        _repair_distributed_choice_slice_source("value = gathered_continuations[i]")


def test_resolve_tokenizer_json_returns_absolute_local_file(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_json.write_text("{}", encoding="utf-8")

    assert _resolve_tokenizer_json(str(tokenizer_dir), None) == str(tokenizer_json.resolve())


def test_greedy_until_retries_cuda_oom_with_smaller_batches(monkeypatch):
    model = SimpleNamespace(config=SimpleNamespace(batch_size=4))
    observed_batch_sizes = []
    releases = []

    def greedy_until(current_model, docs):
        observed_batch_sizes.append(current_model.config.batch_size)
        if current_model.config.batch_size > 1:
            raise RuntimeError("CUDA out of memory")
        return docs

    monkeypatch.setattr(
        "it_eval_framework.runners.lighteval_entry._release_cuda_memory",
        lambda: releases.append(True),
    )

    assert _run_greedy_until_with_oom_backoff(model, ["request"], greedy_until) == ["request"]
    assert observed_batch_sizes == [4, 2, 1]
    assert model.config.batch_size == 4
    assert len(releases) == 4


def test_greedy_until_does_not_hide_non_oom_errors(monkeypatch):
    model = SimpleNamespace(config=SimpleNamespace(batch_size=4))
    monkeypatch.setattr(
        "it_eval_framework.runners.lighteval_entry._release_cuda_memory",
        lambda: None,
    )

    def greedy_until(current_model, docs):
        raise RuntimeError("unrelated generation failure")

    with pytest.raises(RuntimeError, match="unrelated generation failure"):
        _run_greedy_until_with_oom_backoff(model, [], greedy_until)

    assert model.config.batch_size == 4


def test_distributed_generation_bypasses_and_restores_shared_sample_cache(monkeypatch):
    configured_cache = object()
    model = SimpleNamespace(
        config=SimpleNamespace(batch_size=4),
        accelerator=SimpleNamespace(num_processes=4),
        _cache=configured_cache,
    )
    observed_caches = []
    monkeypatch.setattr(
        "it_eval_framework.runners.lighteval_entry._release_cuda_memory",
        lambda: None,
    )

    def greedy_until(current_model, docs):
        observed_caches.append(current_model._cache)
        return docs

    assert _run_greedy_until_with_oom_backoff(model, ["request"], greedy_until) == ["request"]
    assert observed_caches == [None]
    assert model._cache is configured_cache


def test_single_process_generation_keeps_sample_cache_enabled(monkeypatch):
    configured_cache = object()
    model = SimpleNamespace(
        config=SimpleNamespace(batch_size=4),
        accelerator=SimpleNamespace(num_processes=1),
        _cache=configured_cache,
    )
    monkeypatch.setattr(
        "it_eval_framework.runners.lighteval_entry._release_cuda_memory",
        lambda: None,
    )

    def greedy_until(current_model, docs):
        assert current_model._cache is configured_cache
        return docs

    assert _run_greedy_until_with_oom_backoff(model, ["request"], greedy_until) == ["request"]
    assert model._cache is configured_cache


def test_single_process_generation_can_force_prediction_cache_off(monkeypatch):
    configured_cache = object()
    model = SimpleNamespace(
        config=SimpleNamespace(batch_size=4),
        accelerator=SimpleNamespace(num_processes=1),
        _cache=configured_cache,
    )
    observed_caches = []
    monkeypatch.setenv("IT_EVAL_DISABLE_SAMPLE_CACHE", "1")
    monkeypatch.setattr(
        "it_eval_framework.runners.lighteval_entry._release_cuda_memory",
        lambda: None,
    )

    def greedy_until(current_model, docs):
        observed_caches.append(current_model._cache)
        return docs

    assert _run_greedy_until_with_oom_backoff(model, ["request"], greedy_until) == ["request"]
    assert observed_caches == [None]
    assert model._cache is configured_cache


def test_optional_positive_environment_integer(monkeypatch):
    monkeypatch.delenv("TOKEN_CAP", raising=False)
    assert _optional_positive_environment_integer("TOKEN_CAP") is None

    monkeypatch.setenv("TOKEN_CAP", "64")
    assert _optional_positive_environment_integer("TOKEN_CAP") == 64

    for value in ("0", "-1", "invalid"):
        monkeypatch.setenv("TOKEN_CAP", value)
        with pytest.raises(ValueError, match="must be a positive integer"):
            _optional_positive_environment_integer("TOKEN_CAP")


def test_normalize_generation_requests_caps_and_normalizes():
    @dataclass(frozen=True)
    class Request:
        generation_size: int | None
        stop_sequences: object

    requests = [
        Request(generation_size=None, stop_sequences=("\n",)),
        Request(generation_size=400, stop_sequences=["stop"]),
        Request(generation_size=32, stop_sequences=None),
    ]

    normalized, capped = _normalize_generation_requests(requests, 64)

    assert capped == 2
    assert [request.generation_size for request in normalized] == [64, 64, 32]
    assert normalized[0].stop_sequences == ["\n"]
    assert normalized[1].stop_sequences == ["stop"]
    assert normalized[2].stop_sequences is None


def test_environment_flag_enabled(monkeypatch):
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("CACHE_FLAG", value)
        assert _environment_flag_enabled("CACHE_FLAG") is True

    for value in ("", "0", "false", "NO", "off"):
        monkeypatch.setenv("CACHE_FLAG", value)
        assert _environment_flag_enabled("CACHE_FLAG") is False

    monkeypatch.setenv("CACHE_FLAG", "sometimes")
    with pytest.raises(ValueError, match="must be one of"):
        _environment_flag_enabled("CACHE_FLAG")
