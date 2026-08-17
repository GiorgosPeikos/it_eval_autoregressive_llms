from __future__ import annotations

from it_eval_framework.task_registry import resolve_task_aliases


def limit_label(value: int | None, unit: str) -> str:
    return "all available" if value is None else f"at most {value} {unit}"


def print_evaluation_plan(config) -> None:
    """Print enabled datasets/tasks and the scope of every sample limit."""
    print("[run_all] Evaluation plan:", flush=True)
    if config.lighteval.enabled:
        resolved_tasks = resolve_task_aliases(config.lighteval.task_aliases)
        print(
            f"[run_all]   LightEval: enabled; suite={config.lighteval.suite or 'custom'}; "
            f"tasks={len(resolved_tasks)}; {limit_label(config.lighteval.max_samples, 'samples per task')}",
            flush=True,
        )
        for index, (alias, task) in enumerate(zip(config.lighteval.task_aliases, resolved_tasks), start=1):
            print(f"[run_all]     LightEval task {index}/{len(resolved_tasks)}: {alias} -> {task}", flush=True)
    else:
        print("[run_all]   LightEval: disabled", flush=True)
    if config.blimp_it.enabled:
        subset = config.blimp_it.dataset_subset or "all dataset subsets"
        print(
            f"[run_all]   BLiMP-IT: repo={config.blimp_it.dataset_repo}; subset={subset}; "
            f"split={config.blimp_it.split}; {limit_label(config.blimp_it.max_samples, 'examples total')}",
            flush=True,
        )
    if config.perplexity and config.perplexity.enabled:
        dataset = config.perplexity.dataset_repo or config.perplexity.dataset_path
        print(
            f"[run_all]   Perplexity: dataset={dataset}; subset={config.perplexity.dataset_subset}; "
            f"split={config.perplexity.split}; {limit_label(config.perplexity.max_documents, 'documents')}; "
            f"max_tokens_per_document={config.perplexity.max_tokens_per_document or 'all'}",
            flush=True,
        )
    if config.generation.enabled:
        print(
            f"[run_all]   Generation: prompts={config.generation.prompts_path}; "
            f"{limit_label(config.generation.max_prompts, 'prompts')}; profiles={len(config.generation.profiles)}",
            flush=True,
        )
