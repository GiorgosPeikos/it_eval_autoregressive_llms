from __future__ import annotations

import argparse

from it_eval_framework.api import evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="it-eval", description="Evaluate Italian autoregressive language models.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="Run an evaluation from a YAML configuration.")
    evaluate_parser.add_argument("--config", required=True, help="Path to an evaluation YAML file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        print(evaluate(args.config))
