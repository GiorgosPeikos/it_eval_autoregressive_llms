from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path, PureWindowsPath

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


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
        if kwargs.get("use_fast", True) and "tokenizer_file" not in kwargs:
            tokenizer_file = _resolve_tokenizer_json(str(source), kwargs.get("revision"))
            if tokenizer_file:
                kwargs["tokenizer_file"] = tokenizer_file
        return original_from_pretrained(source, *args, **kwargs)

    AutoTokenizer.from_pretrained = compatible_from_pretrained


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
        normalized_docs = []
        for doc in docs:
            stop_sequences = doc.stop_sequences
            if stop_sequences is not None and not isinstance(stop_sequences, list):
                doc = replace(doc, stop_sequences=list(stop_sequences))
            normalized_docs.append(doc)
        return original_greedy_until(self, normalized_docs)

    TransformersModel.greedy_until = patched_greedy_until


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
    _patch_transformers_pmi_loglikelihood()
    _patch_transformers_greedy_until_stop_sequences()
    _patch_problematic_task_revisions()
    from lighteval.__main__ import app

    app()


if __name__ == "__main__":
    main()
