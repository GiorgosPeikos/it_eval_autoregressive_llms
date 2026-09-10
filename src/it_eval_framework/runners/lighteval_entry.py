from __future__ import annotations

import gc
import inspect
import os
import re
import textwrap
from collections import defaultdict
from dataclasses import replace
from functools import update_wrapper
from pathlib import Path, PureWindowsPath

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


_BROKEN_DISTRIBUTED_CHOICE_SLICE = "gathered_continuations[i][:num_choices][:len_choice]"
_FIXED_DISTRIBUTED_CHOICE_SLICE = "gathered_continuations[i][:num_choices, :len_choice]"
_GENERATION_TOKEN_CAP_ENV = "IT_EVAL_GENERATION_TOKEN_CAP"
_DISABLE_SAMPLE_CACHE_ENV = "IT_EVAL_DISABLE_SAMPLE_CACHE"


def _patch_windows_cache_paths() -> None:
    if os.name != "nt":
        return

    from lighteval.utils.cache_management import SampleCache

    original_get_cache_path = SampleCache.get_cache_path

    def sanitized_get_cache_path(self, task_id):
        path = original_get_cache_path(self, task_id)
        return _sanitize_windows_cache_path(path)

    SampleCache.get_cache_path = sanitized_get_cache_path


def _patch_xxhash_string_inputs() -> None:
    import xxhash

    original_xxh64 = xxhash.xxh64

    def compatible_xxh64(data=b"", *args, **kwargs):
        if isinstance(data, str):
            data = data.encode("utf-8")
        return original_xxh64(data, *args, **kwargs)

    xxhash.xxh64 = compatible_xxh64


def _resolve_tokenizer_json(source: str, revision: str | None) -> str | None:
    local_source = Path(source)
    if local_source.exists():
        candidate = local_source / "tokenizer.json" if local_source.is_dir() else local_source
        return str(candidate.resolve()) if candidate.is_file() else None
    try:
        return hf_hub_download(repo_id=source, filename="tokenizer.json", revision=revision)
    except EntryNotFoundError:
        return None


def _patch_transformers_tokenizer_file_resolution() -> None:
    """Give LightEval an absolute fast-tokenizer path on Transformers 4.57."""
    from transformers import AutoTokenizer

    original_from_pretrained = AutoTokenizer.from_pretrained

    def compatible_from_pretrained(source, *args, **kwargs):
        return _compatible_tokenizer_from_pretrained(
            original_from_pretrained,
            source,
            *args,
            **kwargs,
        )

    AutoTokenizer.from_pretrained = compatible_from_pretrained


def _compatible_tokenizer_from_pretrained(original_from_pretrained, source, *args, **kwargs):
    """Load Transformers 5 generic fast-tokenizer exports under 4.57."""
    from transformers import PreTrainedTokenizerFast

    tokenizer_file = kwargs.get("tokenizer_file")
    if kwargs.get("use_fast", True) and tokenizer_file is None:
        tokenizer_file = _resolve_tokenizer_json(str(source), kwargs.get("revision"))
        if tokenizer_file:
            kwargs["tokenizer_file"] = tokenizer_file
    try:
        return original_from_pretrained(source, *args, **kwargs)
    except ValueError as error:
        if (
            not kwargs.get("use_fast", True)
            or tokenizer_file is None
            or "Tokenizer class TokenizersBackend does not exist" not in str(error)
        ):
            raise
        print(
            "[tokenizer] AutoTokenizer cannot resolve TokenizersBackend; "
            "loading tokenizer.json with PreTrainedTokenizerFast.",
            flush=True,
        )
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("use_fast", None)
        return PreTrainedTokenizerFast.from_pretrained(
            source,
            *args,
            **fallback_kwargs,
        )


def _sanitize_filename_component(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "__", value).rstrip(" .")


def _sanitize_windows_cache_path(path: str | Path) -> Path:
    windows_path = PureWindowsPath(path)
    sanitized_parts = []
    for index, part in enumerate(windows_path.parts):
        if index == 0 and windows_path.anchor:
            sanitized_parts.append(part)
            continue
        sanitized_parts.append(_sanitize_filename_component(part))
    return Path(PureWindowsPath(*sanitized_parts))


def _to_windows_extended_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return f"\\\\?\\{resolved}"
    return resolved


def _patch_details_save_paths() -> None:
    if os.name != "nt":
        return

    from lighteval.logging.evaluation_tracker import EvaluationTracker

    original_save_details = EvaluationTracker.save_details

    def sanitized_save_details(self, date_id: str, details_datasets):
        output_dir_details_sub_folder = self._get_details_sub_folder(date_id)
        self.fs.mkdirs(output_dir_details_sub_folder, exist_ok=True)
        logger = __import__("logging").getLogger(__name__)
        logger.info(f"Saving details to {output_dir_details_sub_folder}")
        for task_name, dataset in details_datasets.items():
            sanitized_task_name = _sanitize_filename_component(task_name)
            output_file_details = (
                output_dir_details_sub_folder / f"details_{sanitized_task_name}_{date_id}.parquet"
            )
            output_file_details.parent.mkdir(parents=True, exist_ok=True)
            with open(_to_windows_extended_path(output_file_details), "wb") as f:
                dataset.to_parquet(f)

    EvaluationTracker.save_details = sanitized_save_details


def _merge_unconditioned_response(conditioned_response, unconditioned_response):
    conditioned_logprobs = list(conditioned_response.logprobs)
    unconditioned_logprobs = list(unconditioned_response.logprobs)
    conditioned_response.unconditioned_logprobs = unconditioned_logprobs
    conditioned_response.logprobs = conditioned_logprobs + unconditioned_logprobs
    return conditioned_response


def _token_data_to_host(value):
    """Store response token IDs as the host-side lists declared by ModelResponse."""
    if all(hasattr(value, attribute) for attribute in ("detach", "cpu", "tolist")):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_token_data_to_host(item) for item in value]
    return value


def _patch_model_response_token_storage() -> None:
    """Prevent completed likelihood responses from retaining CUDA tensors."""
    from lighteval.models.model_output import ModelResponse

    original_init = ModelResponse.__init__

    def host_token_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.input_tokens = _token_data_to_host(self.input_tokens)
        self.output_tokens = _token_data_to_host(self.output_tokens)

    ModelResponse.__init__ = host_token_init


def _repair_distributed_choice_slice_source(source: str) -> str:
    """Repair LightEval 0.13's choice/token-axis slicing typo."""
    if _FIXED_DISTRIBUTED_CHOICE_SLICE in source and _BROKEN_DISTRIBUTED_CHOICE_SLICE not in source:
        return source
    occurrences = source.count(_BROKEN_DISTRIBUTED_CHOICE_SLICE)
    if occurrences != 1:
        raise RuntimeError(
            "Unsupported LightEval _loglikelihood_tokens implementation: "
            f"expected one distributed choice-slice pattern, found {occurrences}."
        )
    return source.replace(_BROKEN_DISTRIBUTED_CHOICE_SLICE, _FIXED_DISTRIBUTED_CHOICE_SLICE)


def _patch_lighteval_distributed_choice_slicing() -> None:
    """Keep continuation choice and token axes intact after Accelerate gather."""
    from lighteval.models.transformers.transformers_model import TransformersModel

    original = TransformersModel._loglikelihood_tokens
    if getattr(original, "_it_eval_distributed_choice_slice_fixed", False):
        return

    source = textwrap.dedent(inspect.getsource(original))
    repaired_source = _repair_distributed_choice_slice_source(source)
    if repaired_source == source:
        return

    namespace = {}
    filename = inspect.getsourcefile(original) or "<lighteval-transformers-model>"
    exec(compile(repaired_source, filename, "exec"), original.__globals__, namespace)
    repaired = namespace[original.__name__]
    update_wrapper(repaired, original)
    repaired._it_eval_distributed_choice_slice_fixed = True
    TransformersModel._loglikelihood_tokens = repaired
    print(
        "[lighteval] Patched distributed continuation slicing across choice and token axes.",
        flush=True,
    )


def _patch_transformers_pmi_loglikelihood() -> None:
    from lighteval.models.transformers.transformers_model import TransformersModel

    original_loglikelihood = TransformersModel.loglikelihood

    def patched_loglikelihood(self, docs):
        accelerator = getattr(self, "accelerator", None)
        disable_single_process_accelerator = getattr(accelerator, "num_processes", 1) == 1
        original_cache = getattr(self, "_cache", None)

        try:
            self._cache = None
            if disable_single_process_accelerator:
                self.accelerator = None
            conditioned_responses = original_loglikelihood(self, docs)
        finally:
            self._cache = original_cache
            if disable_single_process_accelerator:
                self.accelerator = accelerator

        docs_requiring_unconditioned = []
        response_indices = []
        for index, (doc, response) in enumerate(zip(docs, conditioned_responses)):
            n_choices = len(doc.choices)
            if doc.unconditioned_query and len(response.logprobs) < n_choices * 2:
                docs_requiring_unconditioned.append(
                    replace(
                        doc,
                        query=doc.unconditioned_query,
                        instruction=None,
                        fewshot_samples=[],
                    )
                )
                response_indices.append(index)

        if not docs_requiring_unconditioned:
            return conditioned_responses

        try:
            self._cache = None
            if disable_single_process_accelerator:
                self.accelerator = None
            unconditioned_responses = original_loglikelihood(self, docs_requiring_unconditioned)
        finally:
            self._cache = original_cache
            if disable_single_process_accelerator:
                self.accelerator = accelerator

        for response_index, unconditioned_response in zip(response_indices, unconditioned_responses):
            conditioned_responses[response_index] = _merge_unconditioned_response(
                conditioned_responses[response_index],
                unconditioned_response,
            )

        return conditioned_responses

    TransformersModel.loglikelihood = patched_loglikelihood


def _patch_transformers_greedy_until_stop_sequences() -> None:
    from lighteval.models.transformers.transformers_model import TransformersModel

    original_greedy_until = TransformersModel.greedy_until

    def patched_greedy_until(self, docs):
        token_cap = _optional_positive_environment_integer(_GENERATION_TOKEN_CAP_ENV)
        normalized_docs, capped_requests = _normalize_generation_requests(docs, token_cap)
        if token_cap is not None:
            print(
                "[lighteval] Applied generation token cap="
                f"{token_cap} to {capped_requests}/{len(normalized_docs)} requests.",
                flush=True,
            )
        return _run_greedy_until_with_oom_backoff(self, normalized_docs, original_greedy_until)

    TransformersModel.greedy_until = patched_greedy_until


def _sampling_method_name(value) -> str:
    return str(getattr(value, "value", value))


def _canonicalize_sampling_docs(sampling_docs):
    """Give every distributed rank identical method and document ordering."""
    method_priority = {
        "LOGPROBS": 0,
        "GENERATIVE": 1,
        "PERPLEXITY": 2,
    }

    def method_key(method):
        name = _sampling_method_name(method)
        return method_priority.get(name, len(method_priority)), name

    def document_key(doc):
        return (
            str(getattr(doc, "task_name", "")),
            str(getattr(doc, "id", "")),
            str(getattr(doc, "query", "")),
        )

    canonical = defaultdict(list)
    for method in sorted(sampling_docs, key=method_key):
        canonical[method] = sorted(sampling_docs[method], key=document_key)
    return canonical


def _patch_pipeline_sampling_order() -> None:
    """Prevent ranks from entering different Accelerate collectives."""
    from lighteval.pipeline import Pipeline

    original_init_tasks_and_requests = Pipeline._init_tasks_and_requests
    if getattr(original_init_tasks_and_requests, "_it_eval_sampling_order_fixed", False):
        return

    def deterministic_init_tasks_and_requests(self, *args, **kwargs):
        result = original_init_tasks_and_requests(self, *args, **kwargs)
        self.sampling_docs = _canonicalize_sampling_docs(self.sampling_docs)
        return result

    deterministic_init_tasks_and_requests._it_eval_sampling_order_fixed = True
    Pipeline._init_tasks_and_requests = deterministic_init_tasks_and_requests
    print(
        "[lighteval] Patched deterministic sampling-method and document ordering.",
        flush=True,
    )


def _normalize_generation_requests(docs, token_cap: int | None):
    normalized_docs = []
    capped_requests = 0
    for doc in docs:
        stop_sequences = doc.stop_sequences
        if stop_sequences is not None and not isinstance(stop_sequences, list):
            doc = replace(doc, stop_sequences=list(stop_sequences))
        if token_cap is not None and (doc.generation_size is None or doc.generation_size > token_cap):
            doc = replace(doc, generation_size=token_cap)
            capped_requests += 1
        normalized_docs.append(doc)
    return normalized_docs, capped_requests


def _release_cuda_memory() -> None:
    """Release tensors from the preceding LightEval sampling phase."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _optional_positive_environment_integer(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer, received {value!r}.") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, received {value!r}.")
    return parsed


def _environment_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; received {value!r}."
    )


def _run_greedy_until_with_oom_backoff(model, docs, greedy_until):
    """Retry only generation with progressively smaller batches after CUDA OOM."""
    configured_batch_size = getattr(model.config, "batch_size", None)
    current_batch_size = configured_batch_size
    accelerator = getattr(model, "accelerator", None)
    distributed = getattr(accelerator, "num_processes", 1) > 1
    cache_disabled_by_environment = _environment_flag_enabled(_DISABLE_SAMPLE_CACHE_ENV)
    has_sample_cache = hasattr(model, "_cache")
    configured_cache = getattr(model, "_cache", None)
    disable_sample_cache = distributed or cache_disabled_by_environment

    # LightEval 0.13's SampleCache writes one shared parquet file per task.
    # Multiple ranks can overwrite that file concurrently, and a follow-up
    # equivalence run can silently replay predictions from an earlier run.
    # Distributed evaluation must therefore gather fresh rank-local responses
    # instead of reading or writing the process-unsafe prediction cache.
    if disable_sample_cache and has_sample_cache:
        model._cache = None
        if distributed:
            print(
                "[lighteval] Disabled the shared prediction cache for distributed generation.",
                flush=True,
            )
        elif cache_disabled_by_environment:
            print(
                f"[lighteval] Disabled the prediction cache because {_DISABLE_SAMPLE_CACHE_ENV}=1.",
                flush=True,
            )

    try:
        while True:
            _release_cuda_memory()
            try:
                return greedy_until(model, docs)
            except RuntimeError as error:
                is_cuda_oom = "CUDA out of memory" in str(error)
                can_reduce = isinstance(current_batch_size, int) and current_batch_size > 1
                if not is_cuda_oom or not can_reduce:
                    raise

                next_batch_size = max(1, current_batch_size // 2)
                print(
                    "[lighteval] CUDA OOM during generation; "
                    f"retrying all generative requests with batch_size={next_batch_size} "
                    f"(previously {current_batch_size}).",
                    flush=True,
                )
                current_batch_size = next_batch_size
                model.config.batch_size = current_batch_size
    finally:
        model.config.batch_size = configured_batch_size
        if disable_sample_cache and has_sample_cache:
            model._cache = configured_cache
        _release_cuda_memory()


def _patch_problematic_task_revisions() -> None:
    from lighteval.tasks.multilingual.tasks.mlmm_arc_challenge import TASKS_TABLE as arc_tasks
    from lighteval.tasks.multilingual.tasks.mlmm_hellaswag import TASKS_TABLE as hellaswag_tasks

    for task_config in hellaswag_tasks:
        task_config.hf_revision = None

    for task_config in arc_tasks:
        task_config.hf_revision = None


def main() -> None:
    _patch_windows_cache_paths()
    _patch_xxhash_string_inputs()
    _patch_transformers_tokenizer_file_resolution()
    _patch_details_save_paths()
    _patch_model_response_token_storage()
    _patch_lighteval_distributed_choice_slicing()
    _patch_pipeline_sampling_order()
    _patch_transformers_pmi_loglikelihood()
    _patch_transformers_greedy_until_stop_sequences()
    _patch_problematic_task_revisions()
    from lighteval.__main__ import app

    app()


if __name__ == "__main__":
    main()
