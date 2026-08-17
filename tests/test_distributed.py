import pytest
import os
from pathlib import Path
import subprocess
import sys

import torch

from it_eval_framework.config import EvaluationConfig, RuntimeConfig
from it_eval_framework.runners.launch import process_count
from it_eval_framework.utils.distributed import DistributedContext, generation_seed


def minimal_payload():
    return {
        "model": {"source": "owner/model"},
        "lighteval": {"enabled": False, "suite": None},
        "blimp_it": {"enabled": False},
        "perplexity": None,
        "generation": {"enabled": False},
    }


def test_rank_ownership_is_disjoint_and_complete():
    contexts = [DistributedContext(True, rank, 3, rank, f"cuda:{rank}") for rank in range(3)]
    assignments = [[index for index in range(10) if context.owns(index)] for context in contexts]

    assert assignments == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]


def test_generation_seed_is_job_stable():
    first = generation_seed(13, "prompt-a", "greedy")

    assert first == generation_seed(13, "prompt-a", "greedy")
    assert first != generation_seed(13, "prompt-a", "sampled")
    assert first != generation_seed(14, "prompt-a", "greedy")


def test_model_parallel_defaults_to_automatic_device_map():
    payload = minimal_payload()
    payload["runtime"] = {"parallelism": "model_parallel"}

    config = EvaluationConfig.model_validate(payload)

    assert config.model.device_map == "auto"


def test_data_parallel_rejects_device_map():
    payload = minimal_payload()
    payload["runtime"] = {"parallelism": "data_parallel"}
    payload["model"]["device_map"] = "auto"

    with pytest.raises(ValueError, match="cannot be combined"):
        EvaluationConfig.model_validate(payload)


def test_num_processes_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        RuntimeConfig(num_processes=0)


def test_explicit_process_count_is_preserved():
    assert process_count(2) == 2


@pytest.mark.skipif(
    not torch.distributed.is_available() or sys.version_info < (3, 10),
    reason="requires the project's supported Python runtime and torch.distributed",
)
def test_two_process_gloo_smoke():
    env = os.environ.copy()
    source_path = str(Path("src").resolve())
    env["PYTHONPATH"] = source_path + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            "2",
            str(Path("tests/distributed_worker.py").resolve()),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
