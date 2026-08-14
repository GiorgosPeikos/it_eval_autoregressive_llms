from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from it_eval_framework.utils.io import read_json


def aggregate(run_dir: Path) -> pd.DataFrame:
    rows = []
    benchmark_path = run_dir / "benchmark_results.json"
    if benchmark_path.exists():
        payload = read_json(benchmark_path)
        rows.append({"component": "lighteval", "metric": "task_count", "value": len(payload["resolved_tasks"])})
    blimp_path = run_dir / "blimp_it_results.json"
    if blimp_path.exists():
        payload = read_json(blimp_path)
        rows.append({"component": "blimp_it", "metric": "overall_accuracy", "value": payload["overall_accuracy"]})
    perplexity_path = run_dir / "perplexity_results.json"
    if perplexity_path.exists():
        payload = read_json(perplexity_path)
        rows.append({"component": "perplexity", "metric": "token_perplexity", "value": payload["token_perplexity"]})
    generations_path = run_dir / "generations.jsonl"
    if generations_path.exists():
        count = sum(1 for _ in generations_path.open("r", encoding="utf-8"))
        rows.append({"component": "generation", "metric": "num_generations", "value": count})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary = aggregate(run_dir)
    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "report.md").write_text(summary.to_markdown(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
