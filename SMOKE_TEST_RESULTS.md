# Smoke Test Results

Date recorded: 2026-08-14

This is the current smoke-test summary. Detailed historical investigation is retained in `AGENT_HANDOFF_STATE.md` and LightEval task classifications are in `LIGHTEVAL_TASK_ISSUES.md`.

## Validated environment

- Windows
- Python 3.12.10 at `C:\venvs\iteval312`
- supported project range: Python 3.10–3.13
- `lighteval==0.13.0`
- `datasets==3.6.0`

## Current results

- unit tests: passing
- default quick path: passing end to end in the validated environment
- BLiMP-IT bounded smoke: passing
- controlled generation bounded smoke: passing
- held-out perplexity bounded smoke: passing
- bounded Windows LightEval suite: passing for every included alias

The supported bounded LightEval suite contains 39 aliases. `mkqa.long_answer` is intentionally excluded because the current probe reports no documents to evaluate.

## Commands

```powershell
C:\venvs\iteval312\Scripts\python.exe -m pytest
C:\venvs\iteval312\Scripts\python.exe -m it_eval_framework.runners.run_perplexity --config configs/italian_perplexity_smoke.yaml
C:\venvs\iteval312\Scripts\python.exe -m it_eval_framework.runners.run_all --config configs/italian_base_quick.yaml
C:\venvs\iteval312\Scripts\python.exe -m it_eval_framework.runners.run_lighteval --config configs/lighteval_verified_windows.yaml
```

The quick configuration is a repository smoke test, not a publication-scale evaluation. It uses a tiny model and strict sample limits. Preserve the generated run directory for its resolved config, environment snapshot, raw results, normalized metrics, and reproducibility metadata.
