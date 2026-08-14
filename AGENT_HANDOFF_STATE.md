# Agent Handoff State

Date: 2026-08-14

> Historical log: earlier steps intentionally preserve failures seen during stabilization. For current status, use the final step in this file, `LIGHTEVAL_TASK_ISSUES.md`, and the README. Earlier blocker lists are not the current repository state.

Purpose: concise working-state summary for future agents. This file is a local handoff note and does not need to be committed.

## Step 1. Repository inspection and packaging cleanup

- Confirmed project structure under `src/it_eval_framework`, `configs/`, `tests/`, and Colab notebooks.
- Identified immediate packaging conflict:
  - `lighteval 0.13.0` requires `datasets>=4.0.0`
  - repo needs `datasets==3.6.0` for the intended Italian task path
- Removed `lighteval` from hard package dependencies in `pyproject.toml`.
- Switched install flow to:
  1. pin `pip<27`, `setuptools<82`, `wheel`
  2. install `lighteval[multilingual]==0.13.0 --no-deps`
  3. install pinned runtime from `constraints/lighteval-python310-313.txt`
  4. install repo editable with `--no-deps`
- Added `pythonpath = ["src"]` to pytest config.

State after step 1:
- Packaging/install flow is resolver-safe.
- Unit tests can import from `src`.
- Repo metadata correctly treats LightEval as a separately managed runtime dependency.

## Step 2. Colab notebook stabilization

- Fixed Colab clone cells to `%cd /content` before deleting and recloning the repo.
- Updated both notebooks to use the resolver-safe install sequence.
- Added GPU inspection cells.
- Added auto-device selection:
  - `MODEL_DEVICE = None` means choose `cuda` if available, else `cpu`.

State after step 2:
- Colab notebooks can be rerun in the same session without breaking `getcwd()`.
- GPU availability is visible to the user before evaluation starts.
- Notebook-generated configs use the detected device.

## Step 3. Runtime visibility improvements

- Added stage progress prints in `run_all.py`.
- Changed `run_lighteval.py` to stream live output instead of hiding everything in captured logs.
- Added explicit LightEval phase messages:
  - loading/downloading
  - preparing/filtering
  - inference
- Added model loading messages in `utils/modeling.py` showing requested and resolved device.

State after step 3:
- Users can see what the pipeline is doing.
- Model/device resolution is visible in both local and Colab runs.
- LightEval output is easier to interpret, though still noisy.

## Step 4. Quick smoke path stabilization

- Removed unstable LightEval tasks from the original quick suite after repeated failures:
  - ARC upstream dataset failure
  - HellaSwag upstream dataset failure
  - SQuAD generative runtime failure in LightEval
- Final decision: quick smoke path should skip LightEval by default.
- Updated:
  - `configs/italian_base_quick.yaml` -> `lighteval.enabled: false`
  - quick Colab notebook -> `ENABLE_LIGHTEVAL = False`
  - README -> documents that quick smoke skips LightEval

State after step 4:
- Default quick smoke path is intended to validate the repo itself, not unstable upstream LightEval integrations.
- LightEval remains part of the project, but not part of the default smoke path.

## Step 5. Local environment setup

- Verified that original local interpreters were:
  - Python 3.14
  - Python 3.9
- Verified that Python 3.14 is not suitable for the pinned stack:
  - actual `datasets 3.6.0` runtime failure observed on 3.14
- Installed Python 3.12.10 side-by-side using `winget`.
- Kept existing 3.14 and 3.9 installations untouched.
- Created a supported local environment at:
  - `C:\venvs\iteval312`

State after step 5:
- There is now a supported local interpreter for the repo.
- Existing Python installations were preserved.

## Step 6. Local quick smoke validation under Python 3.12.10

- Installed supported stack into `C:\venvs\iteval312`.
- Ran unit tests successfully:
  - `6 passed`
- Fixed two repo issues discovered only during local execution:
  - `utils/env.py` no longer hard-requires `lighteval` when LightEval is disabled
  - BLiMP-IT default split changed from `train` to `test`
- Verified that local quick smoke now completes:
  - skips LightEval
  - runs BLiMP-IT
  - runs perplexity
  - runs generation
  - writes summary/report

State after step 6:
- Local supported quick smoke path works end-to-end under Python 3.12.10.
- This is the current reliable local execution baseline.

## Step 7. LightEval task-by-task stabilization path

- Added `configs/lighteval_task_probe.yaml` for one-task probing.
- Added README section describing the LightEval stabilization workflow:
  - probe one task at a time
  - classify each as working / upstream dataset broken / LightEval runtime bug / unsupported
- Fixed `run_lighteval.py` so it no longer assumes the `lighteval` console script is on PATH:
  - falls back to module execution
  - uses `sys.executable` for the fallback

State after step 7:
- Repo now has an explicit LightEval debugging workflow instead of mixing it into the stable quick smoke path.
- LightEval can be probed one task at a time from the supported local environment.

## Step 8. Initial Windows LightEval blocker

Latest local probe:
- task alias: `xcopa.cf`
- resolved task: `xcopa_ita_cf`

Observed behavior:
- task loading works
- dataset download works
- model loading works
- evaluation starts
- failure occurs during LightEval sample-cache write on Windows

Concrete failure:
- LightEval cache path includes task ids like `xcopa_ita_cf|0`
- `|` is invalid in Windows directory names
- result: cache write fails with Windows path error

Interpretation:
- This is not a repo packaging failure
- not a dataset-repo failure
- not a model loading failure
- it is a LightEval portability issue in cache path construction on Windows

State after step 8:
- We now have a precise LightEval blocker for the current local supported environment.
- Next fix should target Windows-safe LightEval cache behavior.

## Step 9. Windows LightEval compatibility patch and first task classification

- Added a repo-side Windows compatibility entrypoint in `src/it_eval_framework/runners/lighteval_entry.py` that now patches three LightEval portability issues at runtime:
  - sample cache paths containing `|`
  - `xxhash` calls receiving Python `str` instead of bytes in details logging
  - details parquet save paths that fail on Windows due to invalid filename characters and long-path handling
- Verified that the probe command now completes successfully on Windows under Python 3.12.10:
  - `C:\venvs\iteval312\Scripts\python.exe -m it_eval_framework.runners.run_lighteval --config configs/lighteval_task_probe.yaml`
- Added focused regression tests for the compatibility helpers:
  - `tests/test_lighteval_entry.py`
- Re-ran unit tests successfully:
  - `8 passed`

Latest local probe result:
- task alias: `xcopa.cf`
- resolved task: `xcopa_ita_cf`
- status: `working in current local Windows environment`

Observed metric output for the tiny smoke model with `max_samples: 2`:
- `acc_norm_token = 1.0`
- `acc_norm = 0.5`

Interpretation:
- the previously identified Windows cache-path blocker is fixed at the repo wrapper layer
- `xcopa.cf` is no longer blocked by Windows portability issues in the current setup
- this does not yet mean the broader LightEval subset is verified; it only establishes one working task in the probe workflow

State after step 9:
- Local Windows LightEval probing is now operational for at least one task.
- The next useful work is no longer cache-path patching; it is systematic task-by-task classification.

## Step 10. Full bounded LightEval task sweep completed

- Added a bounded batch probe runner:
  - `src/it_eval_framework/runners/probe_lighteval_tasks.py`
- Ran the full registered alias set with:
  - `max_samples: 2`
  - one task at a time
  - per-task timeout
- Saved raw probe outputs to:
  - `evaluation_results/lighteval_task_probe_matrix.json`
  - `evaluation_results/lighteval_task_probe_matrix_remaining.json`
- Added a durable Markdown issue tracker:
  - `LIGHTEVAL_TASK_ISSUES.md`

Full alias count:
- `40`

Verified working aliases:
- `xcopa.cf`
- `xcopa.mcf`
- `xcopa.hybrid`
- `xcsqa.mcf`
- `xcodah.cf`
- `xcodah.mcf`
- `xcodah.hybrid`
- `m3exams.mcf`

Observed failure classes after reviewing probe logs:

1. Windows portability bug due to `:` in expanded task names or task variants
- `mlmm_mmlu.cf`
- `mlmm_mmlu.mcf`
- `mlmm_mmlu.hybrid`
- `exams.cf`
- `exams.mcf`
- `exams.hybrid`
- `mlmm_truthfulqa.cf_mc1`
- `mlmm_truthfulqa.cf_mc2`
- `mlmm_truthfulqa.mcf_mc1`
- `mlmm_truthfulqa.mcf_mc2`
- `mlmm_truthfulqa.hybrid_mc1`
- `mlmm_truthfulqa.hybrid_mc2`

Concrete symptom:
- `NotADirectoryError: [WinError 267] The directory name is invalid`
- current cache-path patch handles `|` but not `:` in expanded task names such as subject/task suffixes

2. Upstream dataset broken / generation failure
- `mlmm_hellaswag.cf`
- `mlmm_hellaswag.mcf`
- `mlmm_hellaswag.hybrid`
- `mlmm_arc_challenge.cf`
- `mlmm_arc_challenge.mcf`
- `mlmm_arc_challenge.hybrid`

Concrete symptom:
- `DatasetGenerationError`

3. LightEval runtime bugs
- `xcsqa.cf`
- `xcsqa.hybrid`
- `m3exams.cf`
- `m3exams.hybrid`
- `mkqa.entity`
- `mkqa.short_phrase`
- `mkqa.number`
- `mkqa.number_with_unit`
- `mkqa.date`
- `mkqa.binary`
- `mintaka.default`

Concrete observed symptoms:
- PMI normalization failure:
  - `AssertionError: unconditioned_logprob must be provided for PMI normalization`
- generative batching/runtime failure:
  - `TypeError: can only concatenate list (not "tuple") to list`
- `m3exams` `cf`/`hybrid` also surfaced an `IndexError` during metric handling after the PMI-related path

4. Pinned runtime / dataset compatibility issue
- `squad_it.default`

Concrete symptom:
- `ValueError: Feature type 'List' not found`

5. No-docs task result in current probe
- `mkqa.long_answer`

Concrete symptom:
- `ValueError: Task mkqa_ita:long_answer has no documents to evaluate skipping.`

6. Timeout under bounded probe
- `global_mmlu.mcf`

Concrete behavior:
- exceeded the configured `180s` timeout before finishing the probe

State after step 10:
- The full registered LightEval alias set has now been classified at a bounded probe level on local Windows.
- The current verified working subset is small and concrete.
- The next repo-side fix with the highest leverage is to extend the Windows cache-path sanitization to handle `:` in task-derived cache path components.

## Step 11. First post-sweep fix: Windows `:` cache-path sanitization

- Extended `src/it_eval_framework/runners/lighteval_entry.py` so the Windows LightEval cache-path patch sanitizes all Windows-invalid filename characters in cache path components, not only `|`.
- Re-ran representative `:`-blocked aliases successfully:
  - `mlmm_mmlu.mcf`
  - `exams.mcf`
  - `mlmm_truthfulqa.mcf_mc1`
- Re-ran the full previously Windows-blocked bucket and saved results to:
  - `evaluation_results/lighteval_task_probe_recheck_colon_fix.json`
  - `evaluation_results/lighteval_task_probe_recheck_windows_bucket.json`

Net result of step 11:
- newly working:
  - `mlmm_mmlu.mcf`
  - `exams.cf`
  - `exams.mcf`
  - `exams.hybrid`
  - all six `mlmm_truthfulqa.*` variants
- remaining blocked in the former Windows bucket:
  - `mlmm_mmlu.cf`
  - `mlmm_mmlu.hybrid`

Interpretation:
- the Windows portability fix worked
- the remaining `mlmm_mmlu` blockers are now exposed as the same LightEval PMI-normalization runtime bug already seen in other `cf`/`hybrid` tasks

State after step 11:
- The largest Windows portability bucket has been materially reduced.
- The highest-value next repair target is now the LightEval PMI-normalization failure family.

## Step 12. Second post-sweep fix: PMI loglikelihood augmentation

- Extended `src/it_eval_framework/runners/lighteval_entry.py` again with a repo-side LightEval wrapper for the Transformers backend:
  - when a task document has `unconditioned_query` and the backend only returned conditioned choice logprobs, the wrapper now performs a second unconditioned `loglikelihood` pass and merges those scores back into the response
  - this second pass explicitly bypasses the LightEval sample cache to avoid conditioned/unconditioned cache-key collisions
- Added a focused unit test for the merge helper in:
  - `tests/test_lighteval_entry.py`
- Re-ran unit tests successfully:
  - `9 passed`
- Re-probed the PMI failure family and saved results to:
  - `evaluation_results/lighteval_task_probe_recheck_pmi_family.json`
  - `evaluation_results/lighteval_task_probe_recheck_pmi_family_round2.json`

Net result of step 12:
- newly working:
  - `xcsqa.cf`
  - `xcsqa.hybrid`
- still blocked:
  - `mlmm_mmlu.cf`
  - `mlmm_mmlu.hybrid`
  - `m3exams.cf`
  - `m3exams.hybrid`

Current interpretation of remaining PMI-family blockers:
- the original missing-`unconditioned_logprob` failure was real and is fixed for at least part of the family
- the remaining blocked tasks now fail later in LightEval token normalization with:
  - `IndexError: list index out of range`
- that means the failure class has narrowed from missing PMI inputs to a deeper response-shape/runtime bug in LightEval for these tasks

State after step 12:
- The repo-side wrapper now fixes two concrete LightEval runtime portability problems:
  - missing Windows-safe cache/detail paths
  - missing unconditioned loglikelihood pass for PMI tasks in the Transformers backend
- The remaining unresolved tasks are now concentrated in narrower runtime bugs rather than broad wrapper failures.

## Step 13. Third post-sweep fix: cache bypass for patched loglikelihood path

- The remaining `IndexError` failures in:
  - `mlmm_mmlu.cf`
  - `mlmm_mmlu.hybrid`
  - `m3exams.cf`
  - `m3exams.hybrid`
  turned out not to be a fundamental task-shape failure in the raw model responses.
- Direct task/model/metric checks showed the patched responses had the expected shapes.
- The failing behavior was resolved by disabling the LightEval sample cache inside the patched `TransformersModel.loglikelihood` wrapper path.

Interpretation:
- the cache interaction was producing or reusing inconsistent shapes for the patched conditioned/unconditioned loglikelihood flow
- for the current local Windows stabilization workflow, bypassing the sample cache in the patched loglikelihood path is the narrowest reliable fix

Verification after step 13:
- `pytest` still passes:
  - `9 passed`
- re-probed and confirmed working:
  - `m3exams.cf`
  - `m3exams.hybrid`
  - `mlmm_mmlu.cf`
  - `mlmm_mmlu.hybrid`
- saved results:
  - `evaluation_results/lighteval_task_probe_recheck_cache_disabled.json`

State after step 13:
- the full PMI-related `cf`/`hybrid` failure family addressed in this pass is now working in the bounded probe setup
- the highest-value remaining work has shifted to:
  - `global_mmlu.mcf` timeout classification
  - generative batching failures (`mkqa.*`, `mintaka.default`)
  - upstream dataset failures

## Step 14. Fourth post-sweep fix: stop-sequence normalization for generative tasks

- Reassessed `global_mmlu.mcf` with a longer bounded timeout and verified it works:
  - completed in about `249s`
  - result recorded in `evaluation_results/lighteval_task_probe_global_mmlu_long_timeout.json`
- Investigated generative task failures and confirmed the backend bug:
  - `TransformersModel._padded_greedy_until` concatenated a Python list with `batch[0].stop_sequences`
  - several tasks provided `stop_sequence` as a tuple
- Patched `src/it_eval_framework/runners/lighteval_entry.py` to normalize `stop_sequences` to a list in the wrapper before `greedy_until`
- Re-probed representative and then fuller generative subsets:
  - `evaluation_results/lighteval_task_probe_generative_recheck.json`
  - `evaluation_results/lighteval_task_probe_generative_recheck_round2.json`
  - `evaluation_results/lighteval_task_probe_generative_recheck_full.json`

Net result of step 14:
- newly working:
  - `global_mmlu.mcf`
  - `mkqa.entity`
  - `mkqa.short_phrase`
  - `mkqa.number`
  - `mkqa.number_with_unit`
  - `mkqa.date`
  - `mkqa.binary`
  - `mintaka.default`
- still not probe-runnable:
  - `mkqa.long_answer`
    - current behavior remains: `Task mkqa_ita:long_answer has no documents to evaluate skipping.`

State after step 14:
- the broad generative tuple/list failure family is fixed
- remaining unresolved items are now mostly:
  - upstream dataset failures
  - `squad_it.default` pinned-stack compatibility
  - `mkqa.long_answer` no-docs-in-probe condition

## Current repo state summary

Stable now:
- resolver-safe install flow
- Colab setup and GPU visibility
- local supported environment on Python 3.12.10
- unit tests
- non-LightEval quick smoke path
- Windows LightEval probe path for `xcopa.cf`
- full bounded classification sweep across all registered LightEval aliases
- Windows `:` cache-path handling for LightEval cache files
- repo-side PMI loglikelihood augmentation for Transformer-backed LightEval tasks
- cache bypass for the patched loglikelihood path to avoid broken conditioned/unconditioned cache interactions
- stop-sequence normalization for generative LightEval tasks in the patched wrapper

Bounded support now:
- LightEval remains disabled in the fast default smoke configuration
- the promoted `verified_windows` suite is the supported bounded local Windows path
- all aliases in that suite passed bounded probes; `mkqa.long_answer` is intentionally excluded because it has no probe documents

Current recommended local commands:

```powershell
C:\venvs\iteval312\Scripts\python.exe -m pytest
C:\venvs\iteval312\Scripts\python.exe -m it_eval_framework.runners.run_all --config configs/italian_base_quick.yaml
C:\venvs\iteval312\Scripts\python.exe -m it_eval_framework.runners.run_lighteval --config configs/lighteval_task_probe.yaml
```

## Recommended next task for future agents

Primary next task:
- decide whether to promote the now-large verified working subset into a supported user-facing LightEval config, or to first formalize exclusions for upstream-broken and unsupported tasks

Then:
1. decide policy for `mkqa.long_answer` in bounded probes
2. keep upstream dataset failures (`mlmm_hellaswag.*`, `mlmm_arc_challenge.*`) documented unless there is a narrow reproducible repo-side mitigation
3. keep `squad_it.default` documented as pinned `datasets` compatibility issue unless the repo is willing to change the runtime stack
4. promote only the verified subset into a supported user-facing LightEval config

## Step 15. Final unresolved recheck and supported subset promotion

- Rechecked the last previously unresolved bucket and reviewed:
  - `evaluation_results/lighteval_task_probe_final_unresolved_recheck.json`
- Confirmed newly working under the current wrapper/runtime path:
  - `mlmm_hellaswag.cf`
  - `mlmm_hellaswag.mcf`
  - `mlmm_hellaswag.hybrid`
  - `mlmm_arc_challenge.cf`
  - `mlmm_arc_challenge.mcf`
  - `mlmm_arc_challenge.hybrid`
  - `squad_it.default`
- Confirmed still unsupported for the bounded probe workflow:
  - `mkqa.long_answer`
    - current behavior remains: `Task mkqa_ita:long_answer has no documents to evaluate skipping.`
- Promoted the passing bounded subset into a supported user-facing suite:
  - added `verified_windows` to `src/it_eval_framework/task_registry.py`
  - wired `configs/lighteval_verified_windows.yaml` to that suite
- Updated `README.md` and `LIGHTEVAL_TASK_ISSUES.md` to match the final recheck state.

State after step 15:
- The previously unresolved HellaSwag, ARC, and SQuAD tasks are no longer open blockers in the current local Windows path.
- The supported bounded LightEval path is now explicit and reproducible through `configs/lighteval_verified_windows.yaml`.
- The only remaining intentionally excluded alias from the otherwise verified bounded subset is `mkqa.long_answer`.

## Step 16. Correct strided perplexity accounting

- Fixed overlapping-window perplexity so context tokens are masked and every target token contributes to NLL once.
- Added a regression test covering two overlapping windows.

State after step 16:
- strided perplexity no longer double-counts overlap when `stride < sequence_length`.

## Step 17. Correct normalized checkpoint comparison

- Checkpoint summaries now join on component, task id, few-shot setting, and metric.
- Duplicate identities fail explicitly instead of creating many-to-many rows.

State after step 17:
- normalized LightEval comparisons no longer produce Cartesian task matches.

## Step 18. Expand input reproducibility safeguards

- Added pinned BLiMP-IT revisions and captured subset fingerprints, sizes, and versions.
- Pinned the public tiny model/tokenizer in LightEval configs.
- Added machine-readable warnings for unpinned remote models, datasets, and local artifacts without a supplied digest.

State after step 18:
- `reproducibility.json` explicitly states whether configurable inputs are fully pinned.

## Step 19. Reject empty perplexity evaluations

- Perplexity finalization now raises when zero target tokens were scored.
- Added a regression test for empty input.

State after step 19:
- empty corpora and one-token-only inputs cannot silently report perplexity `1.0`.

## Step 20. Reconcile normalization documentation

- Replaced the stale README limitation with the current raw-plus-normalized LightEval output contract.

State after step 20:
- user-facing documentation matches the implemented result schema.

## Step 21. Expand normalized-result regression coverage

- Added coverage for subject-expanded task ids, aggregate rows, non-zero few-shot settings, absent standard errors, and legacy benchmark payloads.

State after step 21:
- the normalized schema and its backward-compatible aggregation fallback have explicit regression tests.

## Step 22. Distinguish evaluation configs from support YAML

- Added evaluation-config discovery based on the required top-level model section.
- Added a test that loads every discovered evaluation config without misclassifying generation prompts.

State after step 22:
- automated config audits no longer report `generation_prompts.yaml` as an invalid evaluation config.

## Step 23. Normalize every evaluation component

- Added the shared metric-row schema to BLiMP-IT, perplexity, and generation outputs.
- Aggregation consumes normalized rows while retaining legacy fallbacks.

State after step 23:
- all four evaluation families emit compatible normalized metric rows.
