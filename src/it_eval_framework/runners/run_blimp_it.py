from __future__ import annotations

import argparse
from collections import defaultdict

from datasets import get_dataset_config_names, load_dataset

from it_eval_framework.config import load_config
from it_eval_framework.metrics.minimal_pairs import choose_preferred, score_full_sequence
from it_eval_framework.runners.common import mark_finished, mark_started, prepare_run
from it_eval_framework.utils.io import write_json, write_jsonl
from it_eval_framework.utils.modeling import load_model, load_tokenizer


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
    config = load_config(config_path)
    if not config.blimp_it.enabled:
        raise ValueError("BLiMP-IT is disabled in this config.")

    run_dir, state = prepare_run(config)
    if state.is_complete("blimp_it") and not config.output.overwrite:
        return run_dir

    mark_started(run_dir, "blimp_it")
    state.mark("blimp_it", "running")
    model = load_model(config.model)
    tokenizer = load_tokenizer(config.model)
    rows = []
    grouped = defaultdict(lambda: {"correct": 0, "total": 0})
    correct = 0
    total = 0
    subsets = [config.blimp_it.dataset_subset] if config.blimp_it.dataset_subset else get_dataset_config_names(config.blimp_it.dataset_repo)
    remaining = config.blimp_it.max_samples

    for subset in subsets:
        dataset = load_dataset(config.blimp_it.dataset_repo, subset, split=config.blimp_it.split)
        if remaining is not None:
            if remaining <= 0:
                break
            dataset = dataset.select(range(min(len(dataset), remaining)))

        for index, example in enumerate(dataset):
            good_text, bad_text = select_fields(example, config)
            phenomenon = example.get(config.blimp_it.phenomenon_field, subset or "unknown")
            good_score = score_full_sequence(model, tokenizer, good_text, device=config.model.device)
            bad_score = score_full_sequence(model, tokenizer, bad_text, device=config.model.device)
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
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break

    results = {
        "dataset_repo": config.blimp_it.dataset_repo,
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
    write_json(run_dir / "blimp_it_results.json", results)
    write_jsonl(run_dir / "blimp_it_samples.jsonl", rows)
    state.mark("blimp_it", "completed", num_examples=total)
    mark_finished(run_dir, "blimp_it", {"num_examples": total})
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
