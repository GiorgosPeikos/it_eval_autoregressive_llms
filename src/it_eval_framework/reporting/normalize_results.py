from __future__ import annotations

from typing import Any


RESULT_SCHEMA_VERSION = "1.0"


def metric_row(
    component: str,
    task_id: str,
    metric: str,
    value: int | float,
    *,
    stderr: float | None = None,
    sample_count: int | None = None,
    fewshot: int | str | None = None,
    higher_is_better: bool | None = None,
) -> dict[str, Any]:
    return {
        "component": component,
        "task_id": task_id,
        "fewshot": fewshot,
        "metric": metric,
        "value": value,
        "stderr": stderr,
        "sample_count": sample_count,
        "higher_is_better": higher_is_better,
    }


def _task_config(raw_results: dict[str, Any], task_run: str, task_id: str) -> dict[str, Any]:
    configs = raw_results.get("config_tasks", {})
    if task_run in configs:
        return configs[task_run]
    if task_id in configs:
        return configs[task_id]
    for key, value in configs.items():
        if key.endswith(task_run) or key.endswith(task_id):
            return value
    return {}


def _metric_direction(task_config: dict[str, Any], metric_name: str) -> bool | None:
    for metric in task_config.get("metric", task_config.get("metrics", [])):
        if metric.get("metric_name") == metric_name:
            value = metric.get("higher_is_better")
            return value if isinstance(value, bool) else None
    return None


def normalize_lighteval_results(raw_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert LightEval's nested task results to stable, long-form metric rows."""
    rows: list[dict[str, Any]] = []
    for task_run, metrics in sorted(raw_results.get("results", {}).items()):
        task_id, separator, fewshot = task_run.rpartition("|")
        if not separator:
            task_id, fewshot = task_run, None
        task_config = _task_config(raw_results, task_run, task_id)
        sample_count = task_config.get("effective_num_docs")
        if not isinstance(sample_count, int):
            sample_count = None
        for metric, value in sorted(metrics.items()):
            if metric.endswith("_stderr") or not isinstance(value, (int, float)):
                continue
            stderr = metrics.get(f"{metric}_stderr")
            rows.append(metric_row(
                "lighteval",
                task_id,
                metric,
                value,
                stderr=stderr if isinstance(stderr, (int, float)) else None,
                fewshot=int(fewshot) if fewshot and fewshot.isdigit() else fewshot,
                sample_count=sample_count,
                higher_is_better=_metric_direction(task_config, metric),
            ))
    return rows
