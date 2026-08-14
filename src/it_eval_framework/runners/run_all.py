from __future__ import annotations

import argparse
from pathlib import Path

from it_eval_framework.config import load_config
from it_eval_framework.reporting.aggregate_results import aggregate
from it_eval_framework.runners.run_blimp_it import run as run_blimp_it
from it_eval_framework.runners.run_generation import run as run_generation
from it_eval_framework.runners.run_lighteval import run as run_lighteval
from it_eval_framework.runners.run_perplexity import run as run_perplexity
from it_eval_framework.run_layout import build_run_directory


def run(config_path: str) -> Path:
    config = load_config(config_path)
    run_dir = build_run_directory(config)
    if config.lighteval.enabled:
        run_lighteval(config_path)
    if config.blimp_it.enabled:
        run_blimp_it(config_path)
    if config.perplexity and config.perplexity.enabled:
        run_perplexity(config_path)
    if config.generation.enabled:
        run_generation(config_path)
    summary = aggregate(run_dir)
    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "report.md").write_text(summary.to_markdown(index=False), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_dir = run(args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
