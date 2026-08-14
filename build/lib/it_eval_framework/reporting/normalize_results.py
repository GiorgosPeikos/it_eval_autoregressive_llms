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
) -> dict[str, Any]:
    return {
        "component": component,
        "task_id": task_id,
        "fewshot": fewshot,
        "metric": metric,
        "value": value,
        "stderr": stderr,
        "sample_count": sample_count,
    }


def normalize_lighteval_results(raw_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert LightEval's nested task results to stable, long-form metric rows."""
    rows: list[dict[str, Any]] = []
    for task_run, metrics in sorted(raw_results.get("results", {}).items()):
        task_id, separator, fewshot = task_run.rpartition("|")
        if not separator:
            task_id, fewshot = task_run, None
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
            ))
    return rows
