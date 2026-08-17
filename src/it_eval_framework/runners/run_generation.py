from __future__ import annotations

import argparse
import time

import torch

from it_eval_framework.config import GenerationPrompt, load_yaml
from it_eval_framework.metrics.generation_diagnostics import summarize_generation
from it_eval_framework.reporting.normalize_results import RESULT_SCHEMA_VERSION, metric_row
from it_eval_framework.runners.common import load_runner_config, mark_finished, mark_started, prepare_run
from it_eval_framework.utils.io import write_json, write_jsonl
from it_eval_framework.utils.distributed import (
    flatten_gathered,
    generation_seed,
    initialize_distributed,
    local_model_config,
)
from it_eval_framework.utils.modeling import load_model, load_tokenizer, model_input_device, prepare_generation_inputs


def load_prompts(path: str) -> list[GenerationPrompt]:
    payload = load_yaml(path)
    return [GenerationPrompt.model_validate(item) for item in payload["prompts"]]


def run(config_path: str):
    config = load_runner_config(config_path)
    if not config.generation.enabled:
        raise ValueError("Generation is disabled in this config.")

    run_dir, state = prepare_run(config)
    context = initialize_distributed(config.runtime.parallelism, config.model.device)
    if state.is_complete("generation") and not config.output.overwrite:
        return run_dir

    mark_started(run_dir, "generation")
    if context.is_main:
        state.mark("generation", "running")
    model_config = local_model_config(config.model, context)
    tokenizer = load_tokenizer(model_config)
    model = load_model(model_config)
    input_device = model_input_device(model, model_config.device)
    prompts = load_prompts(config.generation.prompts_path)
    if config.generation.max_prompts:
        prompts = prompts[: config.generation.max_prompts]

    rows = []
    job_index = 0
    for prompt in prompts:
        for profile in config.generation.profiles:
            current_job_index = job_index
            job_index += 1
            if not context.owns(current_job_index):
                continue
            encoded = prepare_generation_inputs(tokenizer, prompt.prompt_text, input_device)
            prompt_token_count = encoded["input_ids"].shape[1]
            effective_seed = generation_seed(config.generation.seed, prompt.prompt_id, profile.name)
            torch.manual_seed(effective_seed)
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
                    "effective_random_seed": effective_seed,
                    "job_index": current_job_index,
                    "decoding_profile": profile.model_dump(mode="json"),
                    "generated_text": completion_text,
                    "full_text": full_text,
                    "token_count": int(generated.shape[1] - prompt_token_count),
                    "generation_latency_seconds": latency,
                    **diagnostics,
                }
            )

    gathered = context.gather(rows)
    if not context.is_main:
        context.barrier()
        return run_dir
    rows = flatten_gathered(gathered)
    rows.sort(key=lambda row: row["job_index"])
    for row in rows:
        row.pop("job_index", None)
    write_jsonl(run_dir / "generations.jsonl", rows)
    write_json(
        run_dir / "generation_results.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "num_generations": len(rows),
            "normalized_metrics": [
                metric_row(
                    "generation",
                    "all",
                    "num_generations",
                    len(rows),
                    sample_count=len(rows),
                    higher_is_better=None,
                )
            ],
        },
    )
    state.mark("generation", "completed", num_generations=len(rows))
    mark_finished(run_dir, "generation", {"num_generations": len(rows)})
    context.barrier()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
