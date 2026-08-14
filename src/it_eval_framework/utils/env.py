from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import datasets
import lighteval
import torch
import transformers


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(workdir: str | Path) -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            capture_output=True,
            text=True,
            check=True,
        )
        return output.stdout.strip() or None
    except Exception:
        return None


def environment_snapshot(workdir: str | Path) -> dict:
    return {
        "timestamp_utc": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit(workdir),
        "lighteval_version": lighteval.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "datasets_version": datasets.__version__,
    }
