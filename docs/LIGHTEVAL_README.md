# Running LightEval reliably

This guide describes the supported LightEval 0.13.0 path for the Italian task registry. LightEval is optional and has a stricter dependency stack than the repository-owned BLiMP-IT, perplexity, and generation runners.

## Start with a LightEval-only smoke test

In either Colab notebook, use:

```python
ENABLE_LIGHTEVAL = True
LIGHTEVAL_SUITE = "quick"
MAX_LIGHTEVAL_SAMPLES = 2

ENABLE_BLIMP_IT = False
ENABLE_PERPLEXITY = False
ENABLE_GENERATION = False
```

Run every cell from the beginning. The notebook installs the pinned environment, performs a LightEval preflight, writes the config, and launches the evaluation. The `quick` suite contains one Italian task (`squad_it.default`), and the two-sample cap verifies model loading, tokenizer loading, dataset preparation, inference, and result normalization without starting a publication-scale run.

Do not set an unbounded sample count until this smoke test completes.

## Installation and preflight

For a repository checkout:

```bash
python -m pip install --upgrade "pip<27" "setuptools<82" wheel
python -m pip install "lighteval[multilingual]==0.13.0" --no-deps
python -m pip install -r constraints/lighteval-python310-313.txt
python -m pip install -e . --no-deps
it-eval-check-lighteval
```

The expected core versions are:

```text
lighteval   0.13.0
datasets    3.6.0
transformers 4.57.1
```

The order is intentional. LightEval 0.13.0 declares a newer `datasets` requirement, while several multilingual Italian task definitions still need the legacy dataset-script behavior provided by `datasets==3.6.0`. Installing LightEval normally and allowing pip to upgrade this dependency can make those tasks unusable.

The runner now always enters LightEval through the repository compatibility layer on Linux, Colab, and Windows. That layer handles:

- absolute fast-tokenizer file resolution for Transformers 4.57;
- string inputs passed to `xxhash`;
- PMI/unconditioned multiple-choice responses;
- generation stop-sequence normalization;
- known invalid upstream dataset revision pins;
- Windows cache and result path restrictions.

## Choose a suite and budget

| Suite | Task variants | Intended use |
|---|---:|---|
| `quick` | 1 | First end-to-end LightEval check |
| `full` | 33 | Curated broad evaluation |
| `verified_windows` | 39 | Broad set retained after bounded task probes |
| `all` | 39 | Every currently evaluable Italian variant in the registry |

Forty aliases are known to the registry, but `mkqa.long_answer` is excluded from `all` because the pinned task and dataset combination produces zero evaluation documents. It remains available as an explicit custom alias for diagnosis, but it cannot currently produce a metric.

`max_samples` limits examples within every selected task. It does not select tasks:

```yaml
lighteval:
  enabled: true
  suite: all
  max_samples: 10
```

This selects 39 tasks and evaluates at most 10 examples from each task. Setting `max_samples: null` evaluates every available example in every selected task.

Be aware that `max_samples` limits inference work, but Hugging Face may still need to download or prepare a task's complete source split before LightEval can select those examples.

## Recommended progression

Use the same immutable model revision throughout this progression:

1. `quick`, 2 samples.
2. `verified_windows`, 2 samples per task.
3. `verified_windows`, 50–100 samples per task.
4. `all`, bounded to the research budget.
5. `all`, unbounded only for the final publication run.

For a bounded broad run in the notebook:

```python
ENABLE_LIGHTEVAL = True
LIGHTEVAL_SUITE = "all"
MAX_LIGHTEVAL_SAMPLES = 10

ENABLE_BLIMP_IT = False
ENABLE_PERPLEXITY = False
ENABLE_GENERATION = False
SAVE_DETAILS = True
```

For the final run:

```python
MAX_LIGHTEVAL_SAMPLES = None
```

Authenticate first when using gated models or large task sweeps:

```bash
hf auth login
```

In Colab, setting `HF_TOKEN` in the settings cell also authenticates Hub requests.

## Reading progress output

Before LightEval starts, the framework prints every selected alias and resolved task ID:

```text
[run_all] LightEval: enabled; suite=all; tasks=39; at most 10 samples per task
[run_all] LightEval task 1/39: mlmm_hellaswag.cf -> mlmm_hellaswag_ita_cf
...
[lighteval] Runtime versions: {...}
[lighteval] Starting dataset preparation; inference follows as each task becomes ready.
```

Dataset download and `Generating ... split` messages describe preparation. They do not mean model inference has completed. LightEval's progress bars begin scoring after task documents and requests have been constructed.

The resolved task aliases are also preserved in `resolved_config.yaml`, and normalized results are written to `benchmark_results.json`.

## Failure handling

On failure, the runner writes the combined output to `lighteval_stdout.log`, marks the LightEval step as failed in `run_state.json`, and includes the last 40 log lines directly in the raised exception. Correct the cause and rerun the same config; only completed steps are skipped when `output.overwrite` is false.

Common causes:

- **Preflight reports the wrong `datasets` version:** rerun the pinned installation cells from a fresh runtime.
- **Tokenizer reports `No such file or directory`:** ensure the checkout contains the current compatibility runner and rerun from the first notebook cell.
- **Hub 401/403:** authenticate or request access to the model/dataset.
- **Hub 429:** authenticate, wait for the rate-limit window, and retry with cached downloads.
- **CUDA out of memory:** keep `model.batch_size: 1`, reduce model precision or task scope, or use model sharding as described in `MULTI_GPU_README.md`.
- **No documents to evaluate:** verify the exact task alias; `mkqa.long_answer` is a known zero-document variant in the pinned stack.
- **A run appears stuck during split generation:** dataset preparation is CPU and storage bound. Inspect the live download counters before assuming inference has stalled.

## Multiple GPUs

The integrated LightEval subprocess runs once under rank 0 during a repository data-parallel run, avoiding nested distributed process groups. The native BLiMP-IT, perplexity, and generation stages use all launched ranks. For standalone distributed LightEval or vLLM operation, follow `MULTI_GPU_README.md` and keep those raw outputs separate from the framework run directory.
