from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from it_eval_framework.config import load_config
from it_eval_framework.metrics.perplexity import compute_window_nll, finalize_perplexity, sliding_windows
from it_eval_framework.runners.common import mark_finished, mark_started, prepare_run
from it_eval_framework.utils.io import write_json
from it_eval_framework.utils.modeling import load_model, load_tokenizer


def load_text_dataset(config) -> Iterable[dict]:
    ppl = config.perplexity
    if ppl.dataset_path:
        suffix = Path(ppl.dataset_path).suffix.lower()
        if suffix == ".jsonl":
            return load_dataset("json", data_files=ppl.dataset_path, split="train")
        return load_dataset("text", data_files=ppl.dataset_path, split="train")
    return load_dataset(
        ppl.dataset_repo,
        ppl.dataset_subset,
        split=ppl.split or "train",
        revision=ppl.dataset_revision,
        trust_remote_code=ppl.dataset_trust_remote_code,
        streaming=ppl.dataset_streaming,
    )


def run(config_path: str):
    config = load_config(config_path)
    if not config.perplexity or not config.perplexity.enabled:
        raise ValueError("Perplexity is disabled in this config.")

    run_dir, state = prepare_run(config)
    if state.is_complete("perplexity") and not config.output.overwrite:
        return run_dir

    mark_started(run_dir, "perplexity")
    state.mark("perplexity", "running")
    model = load_model(config.model)
    tokenizer = load_tokenizer(config.model)
    dataset = load_text_dataset(config)
    try:
        num_rows_before_limit = len(dataset)
    except TypeError:
        num_rows_before_limit = None
    dataset_metadata = {
        "fingerprint": getattr(dataset, "_fingerprint", None),
        "num_rows_before_limit": num_rows_before_limit,
        "dataset_name": getattr(dataset.info, "dataset_name", None),
        "config_name": getattr(dataset.info, "config_name", None),
        "version": str(dataset.info.version) if getattr(dataset.info, "version", None) else None,
    }
    if config.perplexity.max_documents:
        if hasattr(dataset, "select"):
            dataset = dataset.select(range(min(len(dataset), config.perplexity.max_documents)))
        else:
            dataset = islice(dataset, config.perplexity.max_documents)

    per_document = []
    total_nll = 0.0
    total_tokens = 0

    for index, row in enumerate(dataset):
        text = row[config.perplexity.text_field]
        if not isinstance(text, str) or not text.strip():
            continue
        if config.perplexity.add_bos_token and tokenizer.bos_token:
            text = tokenizer.bos_token + text
        if config.perplexity.add_eos_token and tokenizer.eos_token:
            text = text + tokenizer.eos_token
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        if config.perplexity.max_tokens_per_document:
            input_ids = input_ids[:, : config.perplexity.max_tokens_per_document]

        doc_nll = 0.0
        doc_tokens = 0
        for start, end in sliding_windows(input_ids.shape[1], config.perplexity.sequence_length, config.perplexity.stride):
            stat = compute_window_nll(model, input_ids, start, end, device=config.model.device)
            doc_nll += stat.negative_log_likelihood
            doc_tokens += stat.token_count
            if config.perplexity.preserve_document_boundaries and end == input_ids.shape[1]:
                break

        total_nll += doc_nll
        total_tokens += doc_tokens
        if config.perplexity.per_document_stats:
            per_document.append(
                {
                    "document_index": index,
                    "token_count": doc_tokens,
                    "negative_log_likelihood": doc_nll,
                }
            )

    results = finalize_perplexity(total_nll, total_tokens)
    results.update(
        {
            "schema_version": "1.0",
            "dataset_path": config.perplexity.dataset_path,
            "dataset_repo": config.perplexity.dataset_repo,
            "dataset_subset": config.perplexity.dataset_subset,
            "dataset_revision": config.perplexity.dataset_revision,
            "dataset_trust_remote_code": config.perplexity.dataset_trust_remote_code,
            "dataset_streaming": config.perplexity.dataset_streaming,
            "split": config.perplexity.split,
            "text_field": config.perplexity.text_field,
            "dataset_metadata": dataset_metadata,
            "max_documents": config.perplexity.max_documents,
            "max_tokens_per_document": config.perplexity.max_tokens_per_document,
            "sequence_length": config.perplexity.sequence_length,
            "stride": config.perplexity.stride,
            "preserve_document_boundaries": config.perplexity.preserve_document_boundaries,
            "add_bos_token": config.perplexity.add_bos_token,
            "add_eos_token": config.perplexity.add_eos_token,
            "contamination_warning": config.perplexity.contamination_warning,
            "per_document": per_document if config.perplexity.per_document_stats else None,
        }
    )
    write_json(run_dir / "perplexity_results.json", results)
    state.mark("perplexity", "completed", total_token_count=total_tokens)
    mark_finished(run_dir, "perplexity", {"total_token_count": total_tokens})
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
