from __future__ import annotations

import argparse
from collections import defaultdict

from datasets import get_dataset_config_names, load_dataset

from it_eval_framework.metrics.minimal_pairs import choose_preferred, score_full_sequence
from it_eval_framework.reporting.normalize_results import RESULT_SCHEMA_VERSION, metric_row
from it_eval_framework.runners.common import load_runner_config, mark_finished, mark_started, prepare_run
from it_eval_framework.utils.io import write_json, write_jsonl
from it_eval_framework.utils.modeling import load_model, load_tokenizer, model_input_device
from it_eval_framework.utils.distributed import flatten_gathered, initialize_distributed, local_model_config


def select_fields(example: dict, config) -> tuple[str, str]:
    good = example.get(config.blimp_it.good_field)
    bad = example.get(config.blimp_it.bad_field)
    if good and bad:
        return good, bad
    candidates = config.blimp_it.text_fallback_fields
    values = [example.get(name) for name in candidates if example.get(name)]
    if len(values) >= 2:
        return values[0], values[1]
    raise KeyError(f"Could not locate BLiMP-IT sentence fields in example keys: {list(example.keys())}")


def run(config_path: str):
    config = load_runner_config(config_path)
    if not config.blimp_it.enabled:
        raise ValueError("BLiMP-IT is disabled in this config.")

    run_dir, state = prepare_run(config)
    context = initialize_distributed(config.runtime.parallelism, config.model.device)
    if state.is_complete("blimp_it") and not config.output.overwrite:
        return run_dir

    mark_started(run_dir, "blimp_it")
    if context.is_main:
        state.mark("blimp_it", "running")
    model_config = local_model_config(config.model, context)
    model = load_model(model_config)
    tokenizer = load_tokenizer(model_config)
    input_device = model_input_device(model, model_config.device)
    rows = []
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    total = 0
    subsets = (
        [config.blimp_it.dataset_subset]
        if config.blimp_it.dataset_subset
        else get_dataset_config_names(config.blimp_it.dataset_repo, revision=config.blimp_it.dataset_revision)
    )
    remaining = config.blimp_it.max_samples
    global_index = 0
    dataset_metadata = []

    for subset in subsets:
        dataset = load_dataset(
            config.blimp_it.dataset_repo,
            subset,
            split=config.blimp_it.split,
            revision=config.blimp_it.dataset_revision,
            trust_remote_code=config.blimp_it.dataset_trust_remote_code,
        )
        dataset_metadata.append(
            {
                "subset": subset,
                "fingerprint": getattr(dataset, "_fingerprint", None),
                "num_rows": len(dataset),
                "version": str(dataset.info.version) if dataset.info.version else None,
            }
        )
        if remaining is not None:
            if remaining <= 0:
                break
            dataset = dataset.select(range(min(len(dataset), remaining)))

        for index, example in enumerate(dataset):
            owned = context.owns(global_index)
            global_index += 1
            if remaining is not None:
                remaining -= 1
            if not owned:
                if remaining is not None and remaining <= 0:
                    break
                continue
            good_text, bad_text = select_fields(example, config)
            phenomenon = example.get(config.blimp_it.phenomenon_field, subset or "unknown")
            good_score = score_full_sequence(model, tokenizer, good_text, device=input_device)
            bad_score = score_full_sequence(model, tokenizer, bad_text, device=input_device)
            prediction = choose_preferred(good_score, bad_score, normalization="raw")
            is_correct = prediction == 0
            correct += int(is_correct)
            total += 1
            grouped[phenomenon]["correct"] += int(is_correct)
            grouped[phenomenon]["total"] += 1
            rows.append(
                {
                    "subset": subset,
                    "index": index,
                    "phenomenon": phenomenon,
                    "good_text": good_text,
                    "bad_text": bad_text,
                    "good_total_logprob": good_score.total_logprob,
                    "bad_total_logprob": bad_score.total_logprob,
                    "good_avg_logprob_per_token": good_score.avg_logprob_per_token,
                    "bad_avg_logprob_per_token": bad_score.avg_logprob_per_token,
                    "good_avg_logprob_per_char": good_score.avg_logprob_per_char,
                    "bad_avg_logprob_per_char": bad_score.avg_logprob_per_char,
                    "correct": is_correct,
                }
            )
            if remaining is not None and remaining <= 0:
                break

    gathered = context.gather({"rows": rows, "correct": correct, "total": total, "grouped": dict(grouped)})
    if not context.is_main:
        context.barrier()
        return run_dir
    rows = flatten_gathered([[*part["rows"]] for part in gathered])
    rows.sort(key=lambda row: (str(row["subset"]), row["index"]))
    correct = sum(part["correct"] for part in gathered)
    total = sum(part["total"] for part in gathered)
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})
    for part in gathered:
        for name, values in part["grouped"].items():
            grouped[name]["correct"] += values["correct"]
            grouped[name]["total"] += values["total"]

    results = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset_repo": config.blimp_it.dataset_repo,
        "dataset_revision": config.blimp_it.dataset_revision,
        "dataset_metadata": dataset_metadata,
        "split": config.blimp_it.split,
        "overall_accuracy": correct / max(total, 1),
        "num_examples": total,
        "by_phenomenon": {
            name: {
                "accuracy": values["correct"] / max(values["total"], 1),
                "num_examples": values["total"],
            }
            for name, values in sorted(grouped.items())
        },
    }
    results["normalized_metrics"] = [
        metric_row("blimp_it", "all", "accuracy", results["overall_accuracy"], sample_count=total),
        *[
            metric_row("blimp_it", name, "accuracy", values["accuracy"], sample_count=values["num_examples"])
            for name, values in results["by_phenomenon"].items()
        ],
    ]
    write_json(run_dir / "blimp_it_results.json", results)
    write_jsonl(run_dir / "blimp_it_samples.jsonl", rows)
    state.mark("blimp_it", "completed", num_examples=total)
    mark_finished(run_dir, "blimp_it", {"num_examples": total})
    context.barrier()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
