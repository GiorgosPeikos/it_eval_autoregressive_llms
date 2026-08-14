# Italian Autoregressive LLM Evaluation Framework

<p align="center">
  <a href="https://colab.research.google.com/github/GiorgosPeikos/it_eval_autoregressive_llms/blob/main/notebooks/colab_quickstart.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab - Quickstart"/>
  </a>
  <a href="https://colab.research.google.com/github/GiorgosPeikos/it_eval_autoregressive_llms/blob/main/notebooks/colab_model_eval_template.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab - Model Eval"/>
  </a>
</p>

This repository provides a reproducible evaluation framework for Italian decoder-only Hugging Face language models. It is designed for base models, not instruction-tuned or chat models, so it prioritizes continuation likelihood, conditional log-likelihood, perplexity, and controlled generation.

Licensed under the [Apache License 2.0](LICENSE).

## Choose how to start

### 1. Evaluate a model in Colab

Use the **Model Eval** badge above. Select a GPU, enter a Hugging Face model id or Google Drive checkpoint path, run the bounded evaluation, and download the complete result ZIP.

### 2. Use it from another Python project

```bash
python -m pip install it-eval-framework
it-eval evaluate --model my-org/my-italian-model --preset quick --device auto
```

Or call it from Python:

```python
from it_eval_framework import evaluate

run_dir = evaluate(model="my-org/my-italian-model", preset="quick", device="auto")
```

### 3. Develop the framework

Clone the repository and use the pinned development installation described below.

For model revisions, local checkpoints, preset selection, LightEval installation, result interpretation, and archival guidance, see [Evaluate your Italian language model](docs/USER_GUIDE.md).

Version `0.1.1` is published on [PyPI](https://pypi.org/project/it-eval-framework/). GitHub installation remains available when testing unreleased commits.

## Scope

The framework supports:

- LightEval-backed Italian benchmark execution
- BLiMP-IT minimal-pair evaluation by direct sentence likelihood comparison
- Perplexity evaluation on local files or Hugging Face datasets
- Controlled generation with configurable decoding profiles and diagnostics
- Stable run directories, resumable step state, environment capture, and machine-readable outputs
- Aggregation and checkpoint comparison utilities
- A supported Python entry point: `from it_eval_framework import evaluate`
- A single user-facing command: `it-eval evaluate --config model.yaml`
- Package-owned presets: `quick`, `perplexity`, `verified_windows`, and explicitly unbounded `full`
- A shared normalized metric-row schema across LightEval, BLiMP-IT, perplexity, and generation summaries

## Repository structure

```text
configs/
constraints/
docs/
notebooks/
src/it_eval_framework/
tests/
README.md
SMOKE_TEST_RESULTS.md
```

Core logic lives in `src/it_eval_framework`. The notebooks are intentionally thin launchers, not a second implementation.
The `configs/` directory also contains support YAML such as `generation_prompts.yaml`; programmatic audits should use `discover_evaluation_configs()` instead of treating every YAML file as a full evaluation config.

## Supported runtime target

The intended runtime target for the full framework is:

- Python `3.10` to `3.13`
- `lighteval[multilingual]==0.13.0`
- `datasets==3.6.0`

The dependency pins used for that path are listed in `constraints/lighteval-python310-313.txt`.
GitHub Actions runs compilation and the network-free unit suite on Python 3.12 for every push and pull request.

One upstream packaging caveat currently matters:

- `lighteval 0.13.0` declares `datasets>=4.0.0`
- several Italian tasks still rely on dataset-script behavior that works with `datasets 3.6.0`

Because of that, the working bootstrap sequence installs `lighteval` first and then pins the shared runtime stack to the known-good versions used by this repository.

## Installation

Create a fresh virtual environment and install the pinned stack:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip<27" "setuptools<82" wheel
python -m pip install "lighteval[multilingual]==0.13.0" --no-deps
python -m pip install -r constraints/lighteval-python310-313.txt
python -m pip install -e .[dev] --no-deps
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip<27" "setuptools<82" wheel
python -m pip install "lighteval[multilingual]==0.13.0" --no-deps
python -m pip install -r constraints/lighteval-python310-313.txt
python -m pip install -e .[dev] --no-deps
```

For larger LightEval runs that touch many Hub datasets in one session, authenticate with Hugging Face first to avoid anonymous Hub rate limits:

```powershell
hf auth login
```

Or set a token explicitly:

```powershell
$env:HF_TOKEN="your_token_here"
```

## Start here

If you are new to the repository, use this order:

1. run the quick smoke test
2. point the config at your model
3. run the supported evaluation path that matches your goal

The three most useful entry points are:

- quick smoke test in Colab: `notebooks/colab_quickstart.ipynb`
- real-model evaluation in Colab: `notebooks/colab_model_eval_template.ipynb`
- local model config template: `configs/local_model_example.yaml`
- benchmark reference for paper writing: `docs/TASK_REFERENCE.md`

## Quick start

Run tests first:

```bash
python -m pytest
```

Run the compact evaluation suite:

```bash
python -m it_eval_framework.runners.run_all --config configs/italian_base_quick.yaml
```

Run only the fast streaming Italian perplexity check (one document, at most 64 tokens):

```bash
python -m it_eval_framework.runners.run_perplexity --config configs/italian_perplexity_smoke.yaml
```

This verifies the model/corpus/perplexity pipeline; its score is not statistically meaningful.
The smoke config pins both the tiny model and Italian dataset to immutable Hugging Face commit hashes.

The default quick config currently skips LightEval. Its default perplexity example is Italian-only and uses the validation split of `gsarti/clean_mc4_it` rather than an English convenience corpus.
It limits evaluation to three documents and 256 tokens per document; remove or increase these limits for a real measurement.

Run the full suite:

```bash
python -m it_eval_framework.runners.run_all --config configs/italian_base_full.yaml
```

Run the verified bounded LightEval subset on local Windows:

```bash
python -m it_eval_framework.runners.run_lighteval --config configs/lighteval_verified_windows.yaml
```

On the local Windows path validated on **August 14, 2026**, this full bounded LightEval run was successfully re-verified after authenticated Hugging Face access was configured. If you hit `429 Too Many Requests` from the Hub, authenticate and rerun before treating it as a repo failure.

Run individual stages:

```bash
python -m it_eval_framework.runners.run_lighteval --config configs/italian_base_full.yaml
python -m it_eval_framework.runners.run_blimp_it --config configs/italian_base_full.yaml
python -m it_eval_framework.runners.run_perplexity --config configs/italian_base_full.yaml
python -m it_eval_framework.runners.run_generation --config configs/italian_base_full.yaml
```

## Evaluate your model

The intended user flow is:

1. verify the repository with the quick smoke path
2. copy `configs/local_model_example.yaml`
3. replace `model.source` with your checkpoint path or Hugging Face repo id
4. optionally set `tokenizer_source`, `device`, `dtype`, and `batch_size`
5. choose the evaluation scope you want

For a local checkpoint on Windows PowerShell:

```powershell
Copy-Item configs/local_model_example.yaml configs/my_model_eval.yaml
notepad configs/my_model_eval.yaml
```

Then run one of these:

Stable smoke path:

```powershell
python -m it_eval_framework.runners.run_all --config configs/italian_base_quick.yaml
```

Supported bounded LightEval subset on local Windows:

```powershell
python -m it_eval_framework.runners.run_lighteval --config configs/lighteval_verified_windows.yaml
```

Broader all-in-one evaluation path:

```powershell
python -m it_eval_framework.runners.run_all --config configs/my_model_eval.yaml
```

When you prepare a model-specific config, the most important fields are:

- `model.source`: local checkpoint path or Hugging Face repo id
- `model.tokenizer_source`: tokenizer override when needed
- `model.device`: `cpu`, `cuda:0`, or another explicit device
- `lighteval.suite`: `quick`, `full`, or `verified_windows`
- `perplexity.dataset_path` or `perplexity.dataset_repo`: your held-out corpus
- for this repository's intended use, that held-out corpus should be Italian
- the repository defaults use `gsarti/clean_mc4_it` as an Italian-only Hugging Face example corpus

If you want a reproducible local Windows LightEval path today, prefer `verified_windows` over `full`.

## LightEval stabilization workflow

Use the default quick config for a stable smoke test. Treat LightEval separately until a verified task subset is established.

For one-task LightEval probing, start from:

```bash
python -m it_eval_framework.runners.run_lighteval --config configs/lighteval_task_probe.yaml
```

Edit `configs/lighteval_task_probe.yaml` one task alias at a time. Classify each task as:

- working
- upstream dataset broken
- LightEval runtime bug
- unsupported in the pinned stack

The current supported bounded subset is already promoted in `configs/lighteval_verified_windows.yaml` and in the `verified_windows` suite.

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

## What gets evaluated

The repository evaluates four different kinds of behavior:

- benchmark accuracy through LightEval-backed Italian tasks
- grammatical acceptability through BLiMP-IT minimal pairs
- next-token likelihood on held-out text through perplexity
- held-out perplexity in this repository is intended to be computed on Italian text, not on an English convenience corpus
- open-ended continuation behavior through controlled generation

For the exact task-by-task reference, task aliases, formulations, and paper-facing notes, see [docs/TASK_REFERENCE.md](docs/TASK_REFERENCE.md).

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
| Held-out perplexity | Italian local file or Italian HF dataset | sliding-window NLL | token perplexity | contamination warning required |
| Controlled generation | local prompts | greedy / sampled decoding | diagnostics + human review | not a substitute for human evaluation |

## ItaCoLA status

ItaCoLA is not implemented as a default base-model benchmark. The standard task is supervised acceptability classification, and this framework does not present an ad hoc zero-shot prompt as equivalent. A separate downstream supervised module can be added later.

## Windows notes

Windows is supported, but a few practical issues should be expected:

- long path handling can affect large dependency trees such as `torch`
- Hugging Face caching may fall back from symlinks to regular copies
- PowerShell and Windows path separators need slightly different command examples
- some local Windows Python distributions may make the legacy `datasets 3.6.0` path harder to reproduce than in Linux or Colab

These are operational issues, not framework design constraints.

## Current limitations

- LightEval multilingual tasks require the `multilingual` extra
- some datasets may be gated, rate-limited, or temporarily unavailable
- the default Italian perplexity example uses `gsarti/clean_mc4_it`; remote loader code is controlled explicitly by `perplexity.dataset_trust_remote_code`, so inspect the dataset repository before enabling it for a new source
- the LightEval wrapper preserves the complete raw payload and also emits versioned long-form rows with task id, few-shot setting, metric value, and standard error where available
- LightEval output also records the Hugging Face dataset repository refs present in its isolated cache, providing the commit inventory needed to archive or investigate the run
- `global_mmlu` and `mlmm_mmlu` expand to subject-level tasks, so aggregation is still task-level
- `mkqa.long_answer` is excluded from the supported bounded probe path because the current probe finds no documents to evaluate for `mkqa_ita:long_answer`

## Supported bounded LightEval path

On the local Windows target validated on **2026-08-14** with Python `3.12.10`, the supported LightEval subset is:

- the `verified_windows` suite in [src/it_eval_framework/task_registry.py](C:/Users/User/PycharmProjects/PythonProject/it_eval_autoregressive_llms/src/it_eval_framework/task_registry.py)
- the matching config [configs/lighteval_verified_windows.yaml](C:/Users/User/PycharmProjects/PythonProject/it_eval_autoregressive_llms/configs/lighteval_verified_windows.yaml)

That subset contains all aliases that completed in bounded local probes under the current wrapper path, including:

- `mlmm_hellaswag.*`
- `xcopa.*`
- `xcsqa.*`
- `xcodah.*`
- `global_mmlu.mcf`
- `mlmm_mmlu.*`
- `mlmm_arc_challenge.*`
- `m3exams.*`
- `exams.*`
- `mlmm_truthfulqa.*`
- `squad_it.default`
- `mkqa.entity`, `mkqa.short_phrase`, `mkqa.number`, `mkqa.number_with_unit`, `mkqa.date`, `mkqa.binary`
- `mintaka.default`

Excluded from that supported bounded path:

- `mkqa.long_answer`, because the current bounded probe still raises `Task mkqa_ita:long_answer has no documents to evaluate skipping.`

## Adding a new Italian task

1. Verify the exact task name from the installed LightEval registry for the pinned version
2. Add the alias mapping in `src/it_eval_framework/task_registry.py`
3. Add the alias to the desired suite in `DEFAULT_LIGHTEVAL_SUITES`
4. Update the benchmark table in this README
5. Add a smoke config entry with `max_samples` before enabling it broadly

## Smoke-test status

See `SMOKE_TEST_RESULTS.md` for the latest recorded smoke execution notes. That file documents observed results in a specific environment; it is not a statement that every environment will fail in the same way.

## Recommended next steps

1. run publication-scale evaluations with a model and Italian corpus appropriate to the research question
2. pin `model.revision` and `perplexity.dataset_revision` to immutable commit hashes for archival runs
3. add tokenizer and vocabulary compatibility checks before execution

## Reproducibility outputs

Every run directory contains:

- `run_config.yaml`: the fully resolved configuration
- `environment.json`: Python, package, platform, Git commit, dirty-worktree status, and a diff hash
- `reproducibility.json`: result-schema version, configuration hash, and random seed
- `reproducibility.json` also reports `fully_pinned_inputs` and actionable input-pinning issues
- component metadata with UTC start and completion times
- raw component outputs, plus normalized metric rows for every evaluation component

For an archival run, use Python 3.10–3.13, install the pinned LightEval constraints, set immutable model and dataset revisions, start from a clean Git worktree, and retain the entire run directory. A dirty worktree is recorded, but cannot be reconstructed from the hash alone unless its patch is also preserved.
For local checkpoints, calculate an archive digest and set `model.artifact_sha256`; the framework records and checks for the supplied digest but does not hash multi-gigabyte checkpoints during every run.
