from __future__ import annotations

import json
from pathlib import Path

from it_eval_framework.config import EvaluationConfig
from it_eval_framework.run_layout import build_run_directory
from it_eval_framework.utils.env import environment_snapshot, utc_now
from it_eval_framework.utils.hashing import stable_hash
from it_eval_framework.utils.io import ensure_dir, write_json, write_yaml
from it_eval_framework.utils.run_state import RunState


def reproducibility_issues(config: EvaluationConfig) -> list[str]:
    issues = []
    source = config.model.source
    looks_local = Path(source).exists() or Path(source).drive != "" or source.startswith(("./", "../", "/"))
    if looks_local and not config.model.artifact_sha256:
        issues.append("Local model source has no model.artifact_sha256 content digest.")
    if not looks_local and not config.model.revision:
        issues.append("Remote model source is not pinned with model.revision.")
    if config.blimp_it.enabled and not config.blimp_it.dataset_revision:
        issues.append("BLiMP-IT dataset is not pinned with blimp_it.dataset_revision.")
    if config.perplexity and config.perplexity.enabled and config.perplexity.dataset_repo and not config.perplexity.dataset_revision:
        issues.append("Perplexity dataset is not pinned with perplexity.dataset_revision.")
    return issues


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
            "issues": reproducibility_issues(config),
            "fully_pinned_inputs": not reproducibility_issues(config),
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
