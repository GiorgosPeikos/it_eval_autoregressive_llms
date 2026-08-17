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
from it_eval_framework.utils.io import ensure_dir, write_yaml


def run(config_path: str) -> Path:
    config = load_config(config_path, resolve_revisions=True)
    run_dir = build_run_directory(config)
    ensure_dir(run_dir)
    resolved_config_path = run_dir / "resolved_config.yaml"
    write_yaml(resolved_config_path, config.model_dump(mode="json"))
    config_path = str(resolved_config_path)
    print(f"[run_all] Run directory: {run_dir}", flush=True)
    print(f"[run_all] Model revision: {config.model.revision}", flush=True)
    print(f"[run_all] Tokenizer revision: {config.model.tokenizer_revision}", flush=True)
    if config.lighteval.enabled:
        print("[run_all] Starting LightEval", flush=True)
        run_lighteval(config_path)
        print("[run_all] Finished LightEval", flush=True)
    else:
        print("[run_all] Skipping LightEval", flush=True)
    if config.blimp_it.enabled:
        print("[run_all] Starting BLiMP-IT", flush=True)
        run_blimp_it(config_path)
        print("[run_all] Finished BLiMP-IT", flush=True)
    if config.perplexity and config.perplexity.enabled:
        print("[run_all] Starting perplexity", flush=True)
        run_perplexity(config_path)
        print("[run_all] Finished perplexity", flush=True)
    if config.generation.enabled:
        print("[run_all] Starting generation", flush=True)
        run_generation(config_path)
        print("[run_all] Finished generation", flush=True)
    print("[run_all] Aggregating results", flush=True)
    summary = aggregate(run_dir)
    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "report.md").write_text(summary.to_markdown(index=False), encoding="utf-8")
    print("[run_all] Wrote summary.csv and report.md", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_dir = run(args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
