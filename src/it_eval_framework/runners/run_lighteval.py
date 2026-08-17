from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from it_eval_framework.reporting.normalize_results import RESULT_SCHEMA_VERSION, normalize_lighteval_results
from it_eval_framework.runners.common import load_runner_config, mark_finished, mark_started, prepare_run
from it_eval_framework.task_registry import LIGHTEVAL_VERSION, resolve_task_aliases
from it_eval_framework.utils.io import read_json, write_json
from it_eval_framework.utils.env import huggingface_dataset_revisions
from it_eval_framework.utils.distributed import initialize_distributed


def apply_windows_hf_cache_overrides(env: dict[str, str]) -> None:
    if os.name != "nt":
        return

    cache_root = Path(r"C:\iteval_hf")
    cache_root.mkdir(parents=True, exist_ok=True)

    env.setdefault("HF_HOME", str(cache_root))
    env.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    env.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))


def build_model_args(config) -> str:
    items = {
        "model_name": config.model.source,
        "revision": config.model.revision,
        "tokenizer": config.model.tokenizer_id,
        "dtype": config.model.dtype,
        "device": config.model.device,
        "batch_size": config.model.batch_size,
        "trust_remote_code": str(config.model.trust_remote_code).lower(),
    }
    if config.model.max_model_length:
        items["max_length"] = config.model.max_model_length
    if config.model.device_map is not None:
        items["model_parallel"] = "true"
    return ",".join(f"{key}={value}" for key, value in items.items() if value is not None)


def latest_results_json(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("results*.json"))
    return candidates[-1] if candidates else None


def run(config_path: str) -> Path:
    config = load_runner_config(config_path)
    if not config.lighteval.enabled:
        raise ValueError("LightEval is disabled in this config.")

    run_dir, state = prepare_run(config)
    context = initialize_distributed(config.runtime.parallelism, config.model.device)
    if state.is_complete("lighteval") and not config.output.overwrite:
        return run_dir
    if not context.is_main:
        context.barrier()
        return run_dir

    task_names = resolve_task_aliases(config.lighteval.task_aliases)
    lighteval_dir = run_dir / "lighteval_raw"
    mark_started(run_dir, "lighteval")
    state.mark("lighteval", "running", task_names=task_names)

    base_command = [config.runtime.lighteval_command]
    if os.name == "nt":
        base_command = [sys.executable, "-m", "it_eval_framework.runners.lighteval_entry"]
    elif shutil.which(config.runtime.lighteval_command) is None:
        base_command = [sys.executable, "-m", "lighteval"]
    command = [
        *base_command,
        "accelerate",
        build_model_args(config),
        ",".join(task_names),
        "--load-tasks-multilingual",
        "--output-dir",
        str(lighteval_dir),
        "--dataset-loading-processes",
        str(config.lighteval.dataset_loading_processes),
        "--num-fewshot-seeds",
        str(config.lighteval.num_fewshot_seeds),
    ]
    if config.output.save_details:
        command.append("--save-details")
    if config.lighteval.max_samples:
        command.extend(["--max-samples", str(config.lighteval.max_samples)])
    command.extend(config.lighteval.extra_args)

    env = os.environ.copy()
    # LightEval is a child evaluation backend, not a member of our torchrun
    # group. Prevent it from trying to join the parent's process group.
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    apply_windows_hf_cache_overrides(env)
    print(f"[lighteval] Running {len(task_names)} task(s):", flush=True)
    for index, (alias, task_name) in enumerate(zip(config.lighteval.task_aliases, task_names), start=1):
        print(f"[lighteval]   Task {index}/{len(task_names)}: {alias} -> {task_name}", flush=True)
    print(f"[lighteval] Log file: {run_dir / 'lighteval_stdout.log'}", flush=True)
    print(
        "[lighteval] Phase 1/3: loading task definitions and downloading datasets.",
        flush=True,
    )
    print(
        "[lighteval] Phase 2/3: preparing task documents and filtering benchmark splits.",
        flush=True,
    )
    print(
        "[lighteval] Phase 3/3: running model inference after dataset preparation completes.",
        flush=True,
    )
    output_lines: list[str] = []
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)
        completed_returncode = process.wait()
    combined_output = "".join(output_lines)
    (run_dir / "lighteval_stdout.log").write_text(combined_output, encoding="utf-8")
    (run_dir / "lighteval_stderr.log").write_text(
        "stderr was merged into stdout for live progress visibility.\n",
        encoding="utf-8",
    )
    if completed_returncode != 0:
        raise RuntimeError(
            f"LightEval failed with exit code {completed_returncode}. "
            f"See {run_dir / 'lighteval_stdout.log'} and {run_dir / 'lighteval_stderr.log'}."
        )

    raw_results_path = latest_results_json(lighteval_dir)
    if raw_results_path is None:
        raise FileNotFoundError(f"No LightEval JSON results found under {lighteval_dir}")

    raw_results = read_json(raw_results_path)
    benchmark_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "backend": "lighteval",
        "lighteval_version": LIGHTEVAL_VERSION,
        "resolved_tasks": task_names,
        "results_file": str(raw_results_path),
        "normalized_metrics": normalize_lighteval_results(raw_results),
        "huggingface_dataset_revision_inventory": huggingface_dataset_revisions(env),
        "raw_results": raw_results,
    }
    write_json(run_dir / "benchmark_results.json", benchmark_payload)
    state.mark("lighteval", "completed", results_file=str(raw_results_path))
    mark_finished(run_dir, "lighteval", {"resolved_tasks": task_names})
    context.barrier()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
