from __future__ import annotations

import platform
import subprocess
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import datasets
import torch
import transformers

try:
    import lighteval
except ImportError:  # pragma: no cover - optional dependency in non-LightEval runs
    lighteval = None


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


def git_worktree_state(workdir: str | Path) -> dict:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=workdir, capture_output=True, text=True, check=True
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], cwd=workdir, capture_output=True, check=True
        ).stdout
        return {
            "git_dirty": bool(status.strip()),
            "git_status": status.splitlines(),
            "git_diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
        }
    except Exception:
        return {"git_dirty": None, "git_status": [], "git_diff_sha256": None}


def environment_snapshot(workdir: str | Path) -> dict:
    snapshot = {
        "timestamp_utc": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit(workdir),
        "lighteval_version": lighteval.__version__ if lighteval is not None else None,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "datasets_version": datasets.__version__,
    }
    snapshot.update(git_worktree_state(workdir))
    return snapshot
