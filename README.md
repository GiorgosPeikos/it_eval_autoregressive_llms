# Italian Autoregressive LLM Evaluation Framework

[![Open In Colab - Quickstart](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GiorgosPeikos/it_eval_autoregressive_llms/blob/main/notebooks/colab_quickstart.ipynb)
[![Open In Colab - Model Eval](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GiorgosPeikos/it_eval_autoregressive_llms/blob/main/notebooks/colab_model_eval_template.ipynb)

This repository provides a reproducible evaluation framework for Italian decoder-only Hugging Face language models. It is designed for base models, not instruction-tuned or chat models, so it prioritizes continuation likelihood, conditional log-likelihood, perplexity, and controlled generation.

## Scope

The framework supports:

- LightEval-backed Italian benchmark execution
- BLiMP-IT minimal-pair evaluation by direct sentence likelihood comparison
- Perplexity evaluation on local files or Hugging Face datasets
- Controlled generation with configurable decoding profiles and diagnostics
- Stable run directories, resumable step state, environment capture, and machine-readable outputs
- Aggregation and checkpoint comparison utilities

## Repository structure

```text
configs/
constraints/
notebooks/
src/it_eval_framework/
tests/
README.md
SMOKE_TEST_RESULTS.md
```

Core logic lives in `src/it_eval_framework`. The notebooks are intentionally thin launchers, not a second implementation.

## Supported runtime target

The intended runtime target for the full framework is:

- Python `3.10` to `3.13`
- `lighteval[multilingual]==0.13.0`
- `datasets==3.6.0`

The dependency pins used for that path are listed in `constraints/lighteval-python310-313.txt`.

## Installation

Create a fresh virtual environment and install the pinned stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r constraints/lighteval-python310-313.txt
python -m pip install -e .[dev]
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r constraints/lighteval-python310-313.txt
python -m pip install -e .[dev]
```

## Quick start

Run tests first:

```bash
python -m pytest
```

Run the compact evaluation suite:

```bash
python -m it_eval_framework.runners.run_all --config configs/italian_base_quick.yaml
```

Run the full suite:

```bash
python -m it_eval_framework.runners.run_all --config configs/italian_base_full.yaml
```

Run individual stages:

```bash
python -m it_eval_framework.runners.run_lighteval --config configs/italian_base_full.yaml
python -m it_eval_framework.runners.run_blimp_it --config configs/italian_base_full.yaml
python -m it_eval_framework.runners.run_perplexity --config configs/italian_base_full.yaml
python -m it_eval_framework.runners.run_generation --config configs/italian_base_full.yaml
```

## Colab notebooks

The repository includes two Colab-oriented notebooks in `notebooks/`:

- `colab_quickstart.ipynb`
- `colab_model_eval_template.ipynb`

Use them for:

- quick smoke tests on a small public causal LM
- early validation of a Hugging Face repo ID
- interactive setup before moving to a normal YAML config and CLI run

The recommended workflow is:

1. validate the environment in Colab with the quick notebook
2. run the quick config on a small model
3. move to a normal config file for repeated evaluations

## User entry point

The user-facing model entry is always one of:

- a local Hugging Face checkpoint path in `model.source`
- a Hugging Face repo ID in `model.source`

Optional tokenizer overrides are supported with `model.tokenizer_source`.

## Verified LightEval version and Italian task IDs

The framework is pinned to `lighteval==0.13.0`. The Italian task registry was verified against the installed multilingual registry on **2026-08-14**.

Key verified IDs include:

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
| MLMM ARC IT | LightEval / `okapi_arc_challenge` | cf, mcf, hybrid | task metrics from LightEval | task prefix is `mlmm_arc_ita_*` |
| SQuAD-it | LightEval / `crux82/squad_it` | generative QA | task metrics from LightEval | task ID is `squad_ita` |
| MKQA IT | LightEval / `apple/mkqa` | answer-type specific QA | task metrics from LightEval | split by answer class |
| TruthfulQA IT | LightEval / `okapi_truthfulqa` | cf, mcf, hybrid | task metrics from LightEval | mc1/mc2 variants kept separate |
| Held-out perplexity | local file or HF dataset | sliding-window NLL | token perplexity | contamination warning required |
| Controlled generation | local prompts | greedy / sampled decoding | diagnostics + human review | not a substitute for human evaluation |

## ItaCoLA status

ItaCoLA is not implemented as a default base-model benchmark. The standard task is supervised acceptability classification, and this framework does not present an ad hoc zero-shot prompt as equivalent. A separate downstream supervised module can be added later.

## Windows notes

Windows is supported, but a few practical issues should be expected:

- long path handling can affect large dependency trees such as `torch`
- Hugging Face caching may fall back from symlinks to regular copies
- PowerShell and Windows path separators need slightly different command examples

These are operational issues, not framework design constraints.

## Current limitations

- LightEval multilingual tasks require the `multilingual` extra
- some datasets may be gated, rate-limited, or temporarily unavailable
- the current LightEval wrapper preserves raw results and verified task IDs, but does not yet normalize every LightEval metric into a richer per-task schema
- `global_mmlu` and `mlmm_mmlu` expand to subject-level tasks, so aggregation is still task-level

## Adding a new Italian task

1. Verify the exact task name from the installed LightEval registry for the pinned version
2. Add the alias mapping in `src/it_eval_framework/task_registry.py`
3. Add the alias to the desired suite in `DEFAULT_LIGHTEVAL_SUITES`
4. Update the benchmark table in this README
5. Add a smoke config entry with `max_samples` before enabling it broadly

## Smoke-test status

See `SMOKE_TEST_RESULTS.md` for the latest recorded smoke execution notes. That file documents observed results in a specific environment; it is not a statement that every environment will fail in the same way.

## Recommended next steps

1. run the quick suite in Colab or another Python `3.10` to `3.13` environment
2. inspect the actual LightEval output structure from a successful run
3. tighten the LightEval result parser into a normalized per-task schema
4. add dataset metadata capture: split, revision, license, and auth requirements
5. add tokenizer and vocabulary compatibility checks before execution
6. add explicit comparison warnings when task or prompt settings differ across runs
