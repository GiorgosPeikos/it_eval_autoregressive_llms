# Smoke Test Results

Date: 2026-08-14

## Executed in this workspace

- Unit tests: `6 passed`
- Controlled generation runner: `passed`
- LightEval runner: `blocked`
- BLiMP-IT runner: `blocked`
- Perplexity runner on Hugging Face dataset input: `blocked`

## What passed

### Unit tests

Executed with:

```powershell
C:\v\iteval\Scripts\python.exe -m pytest
```

Result:

- `tests/test_config.py`
- `tests/test_generation_diagnostics.py`
- `tests/test_minimal_pairs.py`
- `tests/test_perplexity.py`

All passed on 2026-08-14.

### Controlled generation

Executed with:

```powershell
C:\v\iteval\Scripts\python.exe -m it_eval_framework.runners.run_generation --config configs\italian_base_quick.yaml
```

Result:

- Output written to `evaluation_results/sshleifer__tiny-gpt2/tiny-gpt2/italian_base_quick_8f990fb8a2a8fb92/generations.jsonl`
- The runner produced generations and diagnostics for the configured prompts and decoding profiles.
- Example diagnostic: greedy decoding on `sshleifer/tiny-gpt2` showed very high repetition, which is expected for that tiny smoke model.

## What blocked full end-to-end smoke execution

### 1. LightEval plus `datasets 5.0.1`

Observed on 2026-08-14 with `lighteval 0.13.0`:

- Some Italian tasks still resolve to dataset scripts such as `okapi_hellaswag.py`.
- `datasets 5.0.1` rejects those script-based loads.

Concrete failure:

- `RuntimeError: Dataset scripts are no longer supported, but found okapi_hellaswag.py`

### 2. LightEval plus `datasets 3.6.0` on Python 3.14

After downgrading `datasets` to restore script support:

- task loading progressed further
- but `datasets 3.6.0` failed on Python 3.14 during hashing/pickling

Concrete failure:

- `TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given`

### 3. Local interpreter availability

This machine exposes:

- Python 3.14
- Python 3.9

But the reproducible target stack for this framework is effectively:

- Python 3.10 to 3.13
- `lighteval 0.13.0`
- `datasets 3.6.0`

Python 3.9 is too old for the modern dependency floor, while Python 3.14 is too new for the legacy `datasets` code path still needed here.

## Practical conclusion

The framework code, configs, registry verification, and unit-tested metrics layer are in place.

A full LightEval smoke run should be repeated in a Python 3.10 to 3.13 environment using:

```powershell
python -m pip install -r constraints/lighteval-python310-313.txt
python -m pip install -e .[dev]
python -m it_eval_framework.runners.run_all --config configs\italian_base_quick.yaml
```
