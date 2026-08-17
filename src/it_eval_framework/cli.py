from __future__ import annotations

import argparse

from it_eval_framework.api import evaluate
from it_eval_framework.presets import PRESET_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="it-eval", description="Evaluate Italian autoregressive language models.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a model or run a YAML configuration.")
    source = evaluate_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Path to an advanced evaluation YAML file.")
    source.add_argument("--model", help="Hugging Face model id or local checkpoint path.")
    evaluate_parser.add_argument("--preset", choices=PRESET_NAMES, default="quick")
    evaluate_parser.add_argument("--tokenizer", help="Tokenizer id/path; defaults to the model source.")
    evaluate_parser.add_argument("--revision", help="Model revision; omitted/main values are resolved to a commit SHA.")
    evaluate_parser.add_argument(
        "--tokenizer-revision",
        help="Tokenizer revision; resolved independently when the tokenizer uses another repository.",
    )
    evaluate_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or an explicit device.")
    evaluate_parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    evaluate_parser.add_argument("--batch-size", type=int, default=1)
    evaluate_parser.add_argument("--output-dir", default="evaluation_results")
    evaluate_parser.add_argument("--artifact-sha256", help="Digest for an archived local checkpoint.")
    evaluate_parser.add_argument(
        "--perplexity-subset",
        help="Dataset configuration/subset, for example tiny, small, medium, large, or full for clean_mc4_it.",
    )
    evaluate_parser.add_argument(
        "--perplexity-max-documents",
        type=int,
        help="Maximum streamed documents to score; omit to use the selected preset's default.",
    )
    evaluate_parser.add_argument(
        "--perplexity-max-tokens-per-document",
        type=int,
        help="Maximum tokens scored from each document; omit to use the selected preset's default.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        if args.config:
            run_dir = evaluate(args.config)
        else:
            run_dir = evaluate(
                model=args.model,
                preset=args.preset,
                tokenizer=args.tokenizer,
                revision=args.revision,
                tokenizer_revision=args.tokenizer_revision,
                device=args.device,
                dtype=args.dtype,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                artifact_sha256=args.artifact_sha256,
                perplexity_subset=args.perplexity_subset,
                perplexity_max_documents=args.perplexity_max_documents,
                perplexity_max_tokens_per_document=args.perplexity_max_tokens_per_document,
            )
        print(run_dir)


if __name__ == "__main__":
    main()
