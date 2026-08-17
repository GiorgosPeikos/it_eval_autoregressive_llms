from __future__ import annotations

from typing import Any


def metric_semantics(component: str, metric: str, higher_is_better: bool | None = None) -> dict[str, Any]:
    name = metric.lower()
    if component == "blimp_it" and name == "accuracy":
        return {
            "range": "0–1",
            "direction": "higher is better",
            "meaning": "Fraction of minimal pairs where the grammatical sentence has at least as much total log-probability as the ungrammatical sentence.",
        }
    if component == "perplexity" and name == "token_perplexity":
        return {
            "range": "≥1",
            "direction": "lower is better",
            "meaning": "exp(total negative log-likelihood / scored target tokens); not an accuracy or percentage.",
        }
    if component == "perplexity" and name == "mean_loss":
        return {
            "range": "≥0",
            "direction": "lower is better",
            "meaning": "Mean negative log-likelihood per scored target token.",
        }
    if component == "generation" and name == "num_generations":
        return {
            "range": "count",
            "direction": "descriptive",
            "meaning": "Number of prompt/profile generations produced; this is coverage, not quality.",
        }
    if "exact_match" in name:
        return {
            "range": "0–1",
            "direction": "higher is better",
            "meaning": "Mean indicator that the normalized generated answer exactly matches an accepted reference.",
        }
    if "f1" in name:
        return {
            "range": "usually 0–1",
            "direction": "higher is better",
            "meaning": "F1 overlap/agreement metric; inspect the task configuration for its tokenization and macro/micro aggregation.",
        }
    if "acc" in name or "accuracy" in name:
        return {
            "range": "0–1",
            "direction": "higher is better",
            "meaning": "Fraction of evaluated examples scored correct under this task's choice scoring and normalization.",
        }
    if "perplexity" in name or name in {"bits_per_byte", "ter", "edit_distance"}:
        return {
            "range": "metric-specific",
            "direction": "lower is usually better",
            "meaning": "Error, distance, or predictive-fit metric; interpret using the task's raw LightEval configuration.",
        }
    direction = "higher is better" if higher_is_better is True else "lower is better" if higher_is_better is False else "metric-specific"
    return {
        "range": "metric-specific",
        "direction": direction,
        "meaning": "Consult the task definition and raw LightEval metric metadata for this metric.",
    }


def annotate_metric_rows(rows):
    """Add concise human-readable semantics to a DataFrame or list of rows."""
    import pandas as pd

    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        return frame
    annotations = []
    for row in frame.to_dict(orient="records"):
        raw_direction = row.get("higher_is_better")
        direction_text = str(raw_direction).lower()
        direction = direction_text == "true" if direction_text in {"true", "false"} else None
        annotations.append(
            metric_semantics(
                str(row.get("component", "")),
                str(row.get("metric", "")),
                direction,
            )
        )
    for column in ("range", "direction", "meaning"):
        frame[column] = [annotation[column] for annotation in annotations]
    return frame
