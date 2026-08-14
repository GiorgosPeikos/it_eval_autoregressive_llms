from __future__ import annotations

import argparse
import time

import torch

from it_eval_framework.config import GenerationPrompt, load_config, load_yaml
from it_eval_framework.metrics.generation_diagnostics import summarize_generation
from it_eval_framework.reporting.normalize_results import RESULT_SCHEMA_VERSION, metric_row
from it_eval_framework.runners.common import mark_finished, mark_started, prepare_run
from it_eval_framework.utils.io import write_json, write_jsonl
from it_eval_framework.utils.modeling import load_model, load_tokenizer


def load_prompts(path: str) -> list[GenerationPrompt]:
    payload = load_yaml(path)
    return [GenerationPrompt.model_validate(item) for item in payload["prompts"]]


def run(config_path: str):
    config = load_config(config_path)
    if not config.generation.enabled:
        raise ValueError("Generation is disabled in this config.")

    run_dir, state = prepare_run(config)
    if state.is_complete("generation") and not config.output.overwrite:
        return run_dir

    mark_started(run_dir, "generation")
    state.mark("generation", "running")
    tokenizer = load_tokenizer(config.model)
    model = load_model(config.model)
    prompts = load_prompts(config.generation.prompts_path)
    if config.generation.max_prompts:
        prompts = prompts[: config.generation.max_prompts]

    torch.manual_seed(config.generation.seed)
    rows = []
    for prompt in prompts:
        encoded = tokenizer(prompt.prompt_text, return_tensors="pt").to(config.model.device)
        prompt_token_count = encoded["input_ids"].shape[1]
        for profile in config.generation.profiles:
            generation_kwargs = {
                "max_new_tokens": profile.max_new_tokens,
                "do_sample": profile.do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if profile.do_sample:
                generation_kwargs["temperature"] = profile.temperature
                generation_kwargs["top_p"] = profile.top_p

            start = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(**encoded, **generation_kwargs)
            latency = time.perf_counter() - start
            full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            completion_text = tokenizer.decode(generated[0][prompt_token_count:], skip_special_tokens=True)
            diagnostics = summarize_generation(completion_text)
            rows.append(
                {
                    "model_identifier": config.model.source,
                    "checkpoint": config.model.revision or "main",
                    "prompt_identifier": prompt.prompt_id,
                    "prompt_category": prompt.category,
                    "prompt_text": prompt.prompt_text,
                    "random_seed": config.generation.seed,
                    "decoding_profile": profile.model_dump(mode="json"),
                    "generated_text": completion_text,
                    "full_text": full_text,
                    "token_count": int(generated.shape[1] - prompt_token_count),
                    "generation_latency_seconds": latency,
                    **diagnostics,
                }
            )

    write_jsonl(run_dir / "generations.jsonl", rows)
    write_json(
        run_dir / "generation_results.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "num_generations": len(rows),
            "normalized_metrics": [
                metric_row("generation", "all", "num_generations", len(rows), sample_count=len(rows))
            ],
        },
    )
    state.mark("generation", "completed", num_generations=len(rows))
    mark_finished(run_dir, "generation", {"num_generations": len(rows)})
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
