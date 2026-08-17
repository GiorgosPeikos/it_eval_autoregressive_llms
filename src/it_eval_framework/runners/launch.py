from __future__ import annotations

import argparse
import subprocess
import sys

import torch

from it_eval_framework.config import load_config


def process_count(configured: int | str) -> int:
    if isinstance(configured, int):
        return configured
    count = torch.cuda.device_count()
    return max(count, 1)


def run(config_path: str, num_processes: int | None = None) -> int:
    config = load_config(config_path)
    mode = config.runtime.parallelism
    count = num_processes or process_count(config.runtime.num_processes)

    if mode in {"single", "model_parallel"}:
        count = 1
    available_gpus = torch.cuda.device_count()
    if count > 1 and count > available_gpus:
        raise RuntimeError(
            f"Requested {count} processes, but only {available_gpus} CUDA devices are visible. "
            "Adjust runtime.num_processes or CUDA_VISIBLE_DEVICES."
        )

    if count == 1:
        command = [sys.executable, "-m", "it_eval_framework.runners.run_all", "--config", config_path]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(count),
            "-m",
            "it_eval_framework.runners.run_all",
            "--config",
            config_path,
        ]
    print(f"[launch] Parallelism: {mode}; processes: {count}", flush=True)
    return subprocess.run(command, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch an evaluation on one or more GPUs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-processes", type=int, default=None)
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.num_processes))


if __name__ == "__main__":
    main()
