# Italian Autoregressive LLM Evaluation Framework

This repository provides a reproducible evaluation stack for Italian decoder-only Hugging Face language models. The framework is designed for base models, not instruction-tuned models, so it prioritizes continuation likelihood, conditional log-likelihood, perplexity, and controlled generation.

## What is implemented

- LightEval-backed Italian benchmark execution for verified `lighteval==0.13.0` task IDs.
- BLiMP-IT minimal-pair evaluation by direct sentence likelihood comparison.
- Perplexity evaluation for local files or Hugging Face datasets.
- Controlled generation with configurable decoding profiles and diagnostics.
- Stable run directories, environment capture, resumable step state, and machine-readable outputs.
- Aggregation and checkpoint comparison utilities.

## Verified LightEval version and Italian task IDs

The framework is pinned to `lighteval==0.13.0`. The Italian task registry was verified locally on **August 14, 2026** from the installed multilingual registry, not inferred from docs.

Key verified IDs:

- `mlmm_hellaswag_ita_cf`, `mlmm_hellaswag_ita_mcf`, `mlmm_hellaswag_ita_hybrid`
- `xcopa_ita_cf`, `xcopa_ita_mcf`, `xcopa_ita_hybrid`
- `xcsqa_ita_cf`, `xcsqa_ita_mcf`, `xcsqa_ita_hybrid`
- `xcodah_ita_cf`, `xcodah_ita_mcf`, `xcodah_ita_hybrid`
- `global_mmlu_all_ita_mcf:<subject>`
- `mlmm_mmlu_ita_cf:<subject>`, `mlmm_mmlu_ita_mcf:<subject>`, `mlmm_mmlu_ita_hybrid:<subject>`
- `mlmm_arc_ita_cf:challenge`, `mlmm_arc_ita_mcf:challenge`, `mlmm_arc_ita_hybrid:challenge`
- `m3exams_ita_cf`, `m3exams_ita_mcf`, `m3exams_ita_hybrid`
- `exams_ita_cf:<subject>`, `exams_ita_mcf:<subject>`, `exams_ita_hybrid:<subject>`
- `mlmm_truthfulqa_ita_cf:mc1`, `mlmm_truthfulqa_ita_cf:mc2`, `mlmm_truthfulqa_ita_mcf:mc1`, `mlmm_truthfulqa_ita_hybrid:mc1`
- `squad_ita`
- `mkqa_ita:entity`, `mkqa_ita:short_phrase`, `mkqa_ita:number`, `mkqa_ita:date`, `mkqa_ita:binary`, `mkqa_ita:long_answer`, `mkqa_ita:number_with_unit`
- `mintaka_ita`

## Installation

On Windows, the original workspace venv hit a `torch` path-length failure. For this framework, use Python `3.10` to `3.13`. Python `3.14` is not compatible with the legacy `datasets` path still required by several LightEval Italian benchmarks.

```powershell
py -3.12 -m venv C:\v\iteval
C:\v\iteval\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
C:\v\iteval\Scripts\python.exe -m pip install -e .[dev]
```

## Quick start

Quick smoke-oriented command:

```powershell
C:\v\iteval\Scripts\python.exe -m it_eval_framework.runners.run_all --config configs\italian_base_quick.yaml
```

## Colab notebooks

Two Colab-oriented notebooks are included in [notebooks](C:/Users/User/PycharmProjects/PythonProject/it_eval_autoregressive_llms/notebooks):

- [colab_quickstart.ipynb](C:/Users/User/PycharmProjects/PythonProject/it_eval_autoregressive_llms/notebooks/colab_quickstart.ipynb): clone, install, write a temporary quick config, run tests, and launch the quick suite
- [colab_model_eval_template.ipynb](C:/Users/User/PycharmProjects/PythonProject/it_eval_autoregressive_llms/notebooks/colab_model_eval_template.ipynb): a thinner launcher for evaluating a real HF repo id or checkpoint path

These notebooks are intentionally thin wrappers. The package and CLI remain the source of truth.

Complete evaluation command:

```powershell
python -m it_eval_framework.runners.run_all --config configs\italian_base_full.yaml
```

Run individual stages:

```powershell
python -m it_eval_framework.runners.run_lighteval --config configs\italian_base_full.yaml
python -m it_eval_framework.runners.run_blimp_it --config configs\italian_base_full.yaml
python -m it_eval_framework.runners.run_perplexity --config configs\italian_base_full.yaml
python -m it_eval_framework.runners.run_generation --config configs\italian_base_full.yaml
```

## User entry point

The user-facing model entry is always one of:

- a local Hugging Face checkpoint path in `model.source`
- a Hugging Face repo ID in `model.source`

Optional tokenizer overrides are supported with `model.tokenizer_source`.

## Output layout

Each run is written under:

```text
evaluation_results/
  <model_name>/
    <checkpoint_name>/
      <run_name>_<config_hash>/
        run_config.yaml
        environment.json
        run_state.json
        benchmark_results.json
        blimp_it_results.json
        perplexity_results.json
        generations.jsonl
        summary.csv
        report.md
```

## Benchmark table

| Benchmark | Source | Formulation | Primary metric | Notes |
|---|---|---|---|---|
| BLiMP-IT | `NeTSlab/BLiMP-IT` | grammatical vs ungrammatical likelihood | pairwise accuracy | direct base-model fit |
| MLMM HellaSwag IT | LightEval / `okapi_hellaswag` | cf, mcf, hybrid | task metrics from LightEval | verified IDs in registry |
| XCOPA IT | LightEval / `cambridgeltl/xcopa` | cf, mcf, hybrid | task metrics from LightEval | verified IDs in registry |
| Global MMLU IT | LightEval / `Global-MMLU` | mcf | task metrics from LightEval | subject-expanded tasks |
| MLMM ARC IT | LightEval / `okapi_arc_challenge` | cf, mcf, hybrid | task metrics from LightEval | task prefix is `mlmm_arc_ita_*`, not `mlmm_arc_challenge_*` |
| SQuAD-it | LightEval / `crux82/squad_it` | generative QA | task metrics from LightEval | task ID is `squad_ita` |
| MKQA IT | LightEval / `apple/mkqa` | answer-type specific QA | task metrics from LightEval | split by answer class |
| TruthfulQA IT | LightEval / `okapi_truthfulqa` | cf, mcf, hybrid | task metrics from LightEval | mc1/mc2 variants kept separate |
| Held-out perplexity | local file or HF dataset | sliding-window NLL | token perplexity | contamination warning required |
| Controlled generation | local prompts | greedy / sampled decoding | diagnostics + human review | not a substitute for human eval |

## ItaCoLA status

ItaCoLA is not implemented as a default base-model benchmark. The standard task is supervised acceptability classification, and this framework does not present an ad hoc zero-shot prompt as equivalent. A downstream supervised module can be added separately.

## Failure handling and limitations

- LightEval multilingual tasks require `lighteval[multilingual]`.
- On this Windows environment, `language_data` was also required for full multilingual registry loading.
- Some datasets may be gated or temporarily unavailable; the runners currently surface those failures directly.
- The current LightEval wrapper preserves raw results and verified IDs, but it does not yet normalize LightEval metrics into a richer per-task schema.
- `global_mmlu` and `mlmm_mmlu` expand to subject-level tasks; aggregation remains task-level.

## Adding a new Italian task

1. Verify the exact task name from the installed LightEval registry for the pinned version.
2. Add the alias mapping in `src/it_eval_framework/task_registry.py`.
3. Add the alias to the desired suite in `DEFAULT_LIGHTEVAL_SUITES`.
4. Update the benchmark table in this README.
5. Add a smoke config entry with `max_samples` before enabling it broadly.

## Smoke-test status

Smoke execution in this workspace was only partial because the machine exposes Python `3.14` and `3.9`, but the full stack needs Python `3.10` to `3.13`. The concrete failures observed here were:

- `datasets 5.0.1` rejects dataset scripts used by some LightEval Italian tasks.
- `datasets 3.6.0` works better for those tasks but fails on Python `3.14` because of a `dill` / `pickle` incompatibility.
- Python `3.9` cannot satisfy the newer `huggingface_hub` and LightEval dependency floor.

## Recommended next steps

1. Run the quick config and inspect the actual LightEval output file structure.
2. Tighten the LightEval result parser into a stable per-task schema.
3. Add dataset metadata capture: split, revision, license, and auth requirements per benchmark.
4. Add tokenizer/model compatibility validation before execution.
5. Add explicit comparison warnings when prompt/task settings differ across runs.
