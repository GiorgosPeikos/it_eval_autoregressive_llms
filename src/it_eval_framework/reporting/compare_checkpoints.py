from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = ["component", "task_id", "fewshot", "metric"]
COMPARABILITY_FIELDS = [
    "lighteval.enabled", "lighteval.suite", "lighteval.task_aliases", "lighteval.max_samples",
    "lighteval.num_fewshot_seeds", "blimp_it.enabled", "blimp_it.dataset_repo",
    "blimp_it.dataset_subset", "blimp_it.dataset_revision", "blimp_it.split", "blimp_it.max_samples",
    "perplexity.enabled", "perplexity.dataset_path", "perplexity.dataset_repo",
    "perplexity.dataset_subset", "perplexity.dataset_revision", "perplexity.split",
    "perplexity.sequence_length", "perplexity.stride", "perplexity.max_documents",
    "perplexity.max_tokens_per_document", "generation.enabled", "generation.prompts_path",
    "generation.seed", "generation.profiles",
]


def _nested(payload: dict, dotted_path: str):
    value = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def comparison_warnings(config_a: dict, config_b: dict, environment_a: dict | None = None, environment_b: dict | None = None) -> list[str]:
    messages = []
    for field in COMPARABILITY_FIELDS:
        if _nested(config_a, field) != _nested(config_b, field):
            messages.append(f"Evaluation setting differs: {field}")
    tokenizer_a = _nested(config_a, "model.tokenizer_source") or _nested(config_a, "model.source")
    tokenizer_b = _nested(config_b, "model.tokenizer_source") or _nested(config_b, "model.source")
    if tokenizer_a != tokenizer_b:
        messages.append("Evaluation setting differs: effective tokenizer")
    for field in ("lighteval_version", "datasets_version", "transformers_version"):
        if environment_a is not None and environment_b is not None and environment_a.get(field) != environment_b.get(field):
            messages.append(f"Environment differs: {field}")
    return messages


def _read_json_if_present(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def compare_summaries(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in IDENTITY_COLUMNS if column in left.columns and column in right.columns]
    if "component" not in keys or "metric" not in keys:
        raise ValueError("Summaries must contain component and metric columns.")
    for name, frame in (("left", left), ("right", right)):
        if frame.duplicated(keys, keep=False).any():
            raise ValueError(f"The {name} summary has duplicate result identities for keys {keys}.")
    left = left.rename(columns={"value": "value_a", "stderr": "stderr_a"})
    right = right.rename(columns={"value": "value_b", "stderr": "stderr_b"})
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one")
    merged["difference"] = merged["value_b"] - merged["value_a"]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-a", required=True)
    parser.add_argument("--summary-b", required=True)
    args = parser.parse_args()
    left = pd.read_csv(args.summary_a)
    right = pd.read_csv(args.summary_b)
    merged = compare_summaries(left, right)
    run_a = Path(args.summary_a).resolve().parent
    run_b = Path(args.summary_b).resolve().parent
    config_a = _read_json_if_present(run_a / "run_config.json")
    config_b = _read_json_if_present(run_b / "run_config.json")
    if config_a is None or config_b is None:
        import yaml
        config_a = yaml.safe_load((run_a / "run_config.yaml").read_text(encoding="utf-8"))
        config_b = yaml.safe_load((run_b / "run_config.yaml").read_text(encoding="utf-8"))
    for message in comparison_warnings(
        config_a,
        config_b,
        _read_json_if_present(run_a / "environment.json"),
        _read_json_if_present(run_b / "environment.json"),
    ):
        warnings.warn(message, stacklevel=1)
    output = Path(args.summary_a).resolve().parent / "comparison.csv"
    merged.to_csv(output, index=False)
    print(output)
