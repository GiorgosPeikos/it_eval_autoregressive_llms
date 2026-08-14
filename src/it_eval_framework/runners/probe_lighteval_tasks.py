from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from it_eval_framework.task_registry import ITALIAN_LIGHTEVAL_TASKS, resolve_task_alias


def flatten_aliases() -> list[str]:
    aliases: list[str] = []
    for group, variants in ITALIAN_LIGHTEVAL_TASKS.items():
        for variant in variants:
            aliases.append(f"{group}.{variant}")
    return aliases


def classify_failure(output: str) -> tuple[str, str]:
    text = output.lower()
    if "datasetgenerationerror" in text:
        return "upstream_dataset_broken", "dataset generation/loading failure"
    if "assertionerror: unconditioned_logprob must be provided for pmi normalization" in text:
        return "lighteval_runtime_bug", "PMI normalization missing unconditioned_logprob"
    if "valueerror: feature type 'list' not found" in text:
        return "upstream_or_pinned_runtime_compat", "datasets feature schema incompatible with pinned stack"
    if "filenotfounderror" in text or "windows path" in text:
        return "windows_portability_bug", "filesystem/path handling failure"
    if "connectionerror" in text or "datasetgenerationerror" in text:
        return "upstream_dataset_broken", "dataset loading/downloading failure"
    if "keyerror" in text and "task" in text:
        return "unsupported", "task resolution failure"
    return "unknown_failure", "unclassified failure"


def run_probe(config_path: Path, alias: str, max_samples: int, timeout_seconds: int) -> dict:
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_config["lighteval"]["task_aliases"] = [alias]
    base_config["lighteval"]["max_samples"] = max_samples

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False, allow_unicode=False)
        temp_config_path = Path(handle.name)

    command = [
        sys.executable,
        "-m",
        "it_eval_framework.runners.run_lighteval",
        "--config",
        str(temp_config_path),
    ]
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        combined_output = (exc.stdout or "") + "\n" + (exc.stderr or "")
        elapsed = round(time.time() - started, 1)
        return {
            "alias": alias,
            "resolved_task": resolve_task_alias(alias),
            "elapsed_seconds": elapsed,
            "returncode": None,
            "status": "timeout",
            "note": f"probe exceeded {timeout_seconds}s timeout",
            "failure_excerpt": combined_output[-4000:],
        }
    finally:
        temp_config_path.unlink(missing_ok=True)

    combined_output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    elapsed = round(time.time() - started, 1)
    result = {
        "alias": alias,
        "resolved_task": resolve_task_alias(alias),
        "elapsed_seconds": elapsed,
        "returncode": completed.returncode,
    }
    if completed.returncode == 0:
        result["status"] = "working"
        result["note"] = "probe completed"
    else:
        status, note = classify_failure(combined_output)
        result["status"] = status
        result["note"] = note
        result["failure_excerpt"] = combined_output[-4000:]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--output-json", default="evaluation_results/lighteval_task_probe_matrix.json")
    parser.add_argument("--aliases", nargs="*", default=None)
    args = parser.parse_args()

    aliases = args.aliases or flatten_aliases()
    config_path = Path(args.config)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total = len(aliases)
    for index, alias in enumerate(aliases, start=1):
        print(f"[probe] {index}/{total} {alias}", flush=True)
        result = run_probe(
            config_path=config_path,
            alias=alias,
            max_samples=args.max_samples,
            timeout_seconds=args.timeout_seconds,
        )
        print(
            f"[probe] {alias} -> {result['status']} ({result['elapsed_seconds']}s)",
            flush=True,
        )
        results.append(result)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
