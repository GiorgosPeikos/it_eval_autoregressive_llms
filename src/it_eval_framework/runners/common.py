from __future__ import annotations

import json
from pathlib import Path

from it_eval_framework.config import EvaluationConfig
from it_eval_framework.run_layout import build_run_directory
from it_eval_framework.utils.env import environment_snapshot, utc_now
from it_eval_framework.utils.hashing import stable_hash
from it_eval_framework.utils.io import ensure_dir, write_json, write_yaml
from it_eval_framework.utils.run_state import RunState


def prepare_run(config: EvaluationConfig) -> tuple[Path, RunState]:
    run_dir = build_run_directory(config)
    ensure_dir(run_dir)
    config_payload = config.model_dump(mode="json")
    write_yaml(run_dir / "run_config.yaml", config_payload)
    write_json(run_dir / "environment.json", environment_snapshot(Path.cwd()))
    write_json(
        run_dir / "reproducibility.json",
        {
            "schema_version": "1.0",
            "config_sha256_prefix": stable_hash(config_payload),
            "seed": config.runtime.seed,
            "run_directory": str(run_dir),
        },
    )
    return run_dir, RunState(run_dir / "run_state.json")


def mark_started(run_dir: Path, step: str) -> None:
    path = run_dir / f"{step}_metadata.json"
    payload = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload["started_at_utc"] = utc_now()
    write_json(path, payload)


def mark_finished(run_dir: Path, step: str, extra: dict | None = None) -> None:
    path = run_dir / f"{step}_metadata.json"
    payload = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload["completed_at_utc"] = utc_now()
    if extra:
        payload.update(extra)
    write_json(path, payload)
