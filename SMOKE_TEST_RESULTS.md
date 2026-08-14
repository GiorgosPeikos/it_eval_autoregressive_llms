# Smoke Test Results

Date recorded: 2026-08-14

This file records observed smoke-test results in one specific development environment. It should be read as an environment report, not as a general statement about the repository.

## Environment summary

Observed local interpreter availability:

- Python 3.14
- Python 3.9

Target runtime for the full framework:

- Python 3.10 to 3.13
- `lighteval[multilingual]==0.13.0`
- `datasets==3.6.0`

## Executed in the recorded environment

- Unit tests: `6 passed`
- Controlled generation runner: `passed`
- LightEval runner: `blocked`
- BLiMP-IT runner: `blocked`
- Perplexity runner on Hugging Face dataset input: `blocked`

## What passed

### Unit tests

Command:

```bash
python -m pytest
```

Result:

- `tests/test_config.py`
- `tests/test_generation_diagnostics.py`
- `tests/test_minimal_pairs.py`
- `tests/test_perplexity.py`

All passed in the recorded environment on 2026-08-14.

### Controlled generation

Command:

```bash
python -m it_eval_framework.runners.run_generation --config configs/italian_base_quick.yaml
```

Result:

- the runner produced `generations.jsonl`
- decoding diagnostics were emitted as expected
- the tiny public smoke model showed high repetition under greedy decoding, which is expected

## What blocked full end-to-end smoke execution

### 1. LightEval plus `datasets 5.0.1`

Observed behavior:

- some Italian tasks still resolve to dataset scripts such as `okapi_hellaswag.py`
- `datasets 5.0.1` rejects those script-based loads

Concrete failure:

- `RuntimeError: Dataset scripts are no longer supported, but found okapi_hellaswag.py`

### 2. LightEval plus `datasets 3.6.0` on Python 3.14

Observed behavior:

- task loading progressed further
- `datasets 3.6.0` then failed on Python 3.14 during hashing and pickling

Concrete failure:

- `TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given`

### 3. Local interpreter availability

Observed behavior:

- Python 3.9 is too old for the full dependency floor used by the intended stack
- Python 3.14 is too new for the legacy `datasets` path still required by several LightEval tasks

## Practical conclusion

The framework code, configs, task registry verification, and unit-tested metrics layer are in place.

A full LightEval smoke run should be repeated in a Python 3.10 to 3.13 environment using:

```bash
python -m pip install -r constraints/lighteval-python310-313.txt
python -m pip install -e .[dev]
python -m it_eval_framework.runners.run_all --config configs/italian_base_quick.yaml
```

## How to use this report

Use this file to understand:

- what was already tested
- what failed in one recorded environment
- why Colab or another clean Linux runtime is a sensible next smoke-test target

Do not treat it as proof that the repository is generally blocked.
