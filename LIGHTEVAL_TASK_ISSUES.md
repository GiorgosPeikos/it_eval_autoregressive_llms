# LightEval Task Issues

Date: 2026-08-14

Purpose: durable issue tracker for bounded LightEval task probes on local Windows with Python 3.12.10 and `max_samples: 2`.

Interpretation:
- `working`: bounded probe completed
- `windows_portability_bug`: repo-side Windows compatibility issue likely fixable here
- `upstream_dataset_broken`: dataset generation/loading failure outside normal repo ownership
- `lighteval_runtime_bug`: LightEval internal/runtime failure after task loading
- `upstream_or_pinned_runtime_compat`: pinned stack incompatibility with current dataset/task
- `timeout`: bounded probe exceeded timeout and needs separate handling

## XCOPA

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `xcopa.cf` | `xcopa_ita_cf` | `working` | probe completed |
| `xcopa.mcf` | `xcopa_ita_mcf` | `working` | probe completed |
| `xcopa.hybrid` | `xcopa_ita_hybrid` | `working` | probe completed |

## XCSQA

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `xcsqa.cf` | `xcsqa_ita_cf` | `working` | probe completed after repo-side PMI loglikelihood augmentation fix |
| `xcsqa.mcf` | `xcsqa_ita_mcf` | `working` | probe completed |
| `xcsqa.hybrid` | `xcsqa_ita_hybrid` | `working` | probe completed after repo-side PMI loglikelihood augmentation fix |

## XCODAH

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `xcodah.cf` | `xcodah_ita_cf` | `working` | probe completed |
| `xcodah.mcf` | `xcodah_ita_mcf` | `working` | probe completed |
| `xcodah.hybrid` | `xcodah_ita_hybrid` | `working` | probe completed |

## MLMM HellaSwag

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `mlmm_hellaswag.cf` | `mlmm_hellaswag_ita_cf` | `working` | probe completed after repo-side task revision override |
| `mlmm_hellaswag.mcf` | `mlmm_hellaswag_ita_mcf` | `working` | probe completed after repo-side task revision override |
| `mlmm_hellaswag.hybrid` | `mlmm_hellaswag_ita_hybrid` | `working` | probe completed after repo-side task revision override |

## Global MMLU

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `global_mmlu.mcf` | `global_mmlu_all_ita_mcf` | `working` | probe completed with a longer bounded timeout (`~249s`) |

## MLMM MMLU

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `mlmm_mmlu.cf` | `mlmm_mmlu_ita_cf` | `working` | probe completed after PMI augmentation and cache bypass in patched loglikelihood path |
| `mlmm_mmlu.mcf` | `mlmm_mmlu_ita_mcf` | `working` | probe completed after Windows `:` cache-path fix |
| `mlmm_mmlu.hybrid` | `mlmm_mmlu_ita_hybrid` | `working` | probe completed after PMI augmentation and cache bypass in patched loglikelihood path |

## MLMM ARC Challenge

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `mlmm_arc_challenge.cf` | `mlmm_arc_ita_cf:challenge` | `working` | probe completed after repo-side task revision override |
| `mlmm_arc_challenge.mcf` | `mlmm_arc_ita_mcf:challenge` | `working` | probe completed after repo-side task revision override |
| `mlmm_arc_challenge.hybrid` | `mlmm_arc_ita_hybrid:challenge` | `working` | probe completed after repo-side task revision override |

## M3ExamS

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `m3exams.cf` | `m3exams_ita_cf` | `working` | probe completed after PMI augmentation and cache bypass in patched loglikelihood path |
| `m3exams.mcf` | `m3exams_ita_mcf` | `working` | probe completed |
| `m3exams.hybrid` | `m3exams_ita_hybrid` | `working` | probe completed after PMI augmentation and cache bypass in patched loglikelihood path |

## Exams

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `exams.cf` | `exams_ita_cf` | `working` | probe completed after Windows `:` cache-path fix |
| `exams.mcf` | `exams_ita_mcf` | `working` | probe completed after Windows `:` cache-path fix |
| `exams.hybrid` | `exams_ita_hybrid` | `working` | probe completed after Windows `:` cache-path fix |

## TruthfulQA

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `mlmm_truthfulqa.cf_mc1` | `mlmm_truthfulqa_ita_cf:mc1` | `working` | probe completed after Windows `:` cache-path fix |
| `mlmm_truthfulqa.cf_mc2` | `mlmm_truthfulqa_ita_cf:mc2` | `working` | probe completed after Windows `:` cache-path fix |
| `mlmm_truthfulqa.mcf_mc1` | `mlmm_truthfulqa_ita_mcf:mc1` | `working` | probe completed after Windows `:` cache-path fix |
| `mlmm_truthfulqa.mcf_mc2` | `mlmm_truthfulqa_ita_mcf:mc2` | `working` | probe completed after Windows `:` cache-path fix |
| `mlmm_truthfulqa.hybrid_mc1` | `mlmm_truthfulqa_ita_hybrid:mc1` | `working` | probe completed after Windows `:` cache-path fix |
| `mlmm_truthfulqa.hybrid_mc2` | `mlmm_truthfulqa_ita_hybrid:mc2` | `working` | probe completed after Windows `:` cache-path fix |

## SQuAD-it

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `squad_it.default` | `squad_ita` | `working` | probe completed in final unresolved recheck under the current wrapper/runtime path |

## MKQA

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `mkqa.entity` | `mkqa_ita:entity` | `working` | probe completed after stop-sequence normalization fix in patched generation path |
| `mkqa.short_phrase` | `mkqa_ita:short_phrase` | `working` | probe completed after stop-sequence normalization fix |
| `mkqa.long_answer` | `mkqa_ita:long_answer` | `unsupported_for_probe` | no docs available in this bounded probe: `Task ... has no documents to evaluate` |
| `mkqa.number` | `mkqa_ita:number` | `working` | probe completed after stop-sequence normalization fix |
| `mkqa.number_with_unit` | `mkqa_ita:number_with_unit` | `working` | probe completed after stop-sequence normalization fix |
| `mkqa.date` | `mkqa_ita:date` | `working` | probe completed after stop-sequence normalization fix |
| `mkqa.binary` | `mkqa_ita:binary` | `working` | probe completed after stop-sequence normalization fix |

## Mintaka

| Alias | Resolved task | Status | Notes |
|---|---|---|---|
| `mintaka.default` | `mintaka_ita` | `working` | probe completed after stop-sequence normalization fix in patched generation path |

## Current Verified Working Subset

- `xcopa.cf`
- `xcopa.mcf`
- `xcopa.hybrid`
- `xcsqa.cf`
- `xcsqa.mcf`
- `xcsqa.hybrid`
- `xcodah.cf`
- `xcodah.mcf`
- `xcodah.hybrid`
- `mlmm_mmlu.mcf`
- `mlmm_mmlu.cf`
- `mlmm_mmlu.hybrid`
- `mlmm_arc_challenge.cf`
- `mlmm_arc_challenge.mcf`
- `mlmm_arc_challenge.hybrid`
- `mlmm_hellaswag.cf`
- `mlmm_hellaswag.mcf`
- `mlmm_hellaswag.hybrid`
- `m3exams.cf`
- `m3exams.mcf`
- `m3exams.hybrid`
- `exams.cf`
- `exams.mcf`
- `exams.hybrid`
- `mlmm_truthfulqa.cf_mc1`
- `mlmm_truthfulqa.cf_mc2`
- `mlmm_truthfulqa.mcf_mc1`
- `mlmm_truthfulqa.mcf_mc2`
- `mlmm_truthfulqa.hybrid_mc1`
- `mlmm_truthfulqa.hybrid_mc2`
- `global_mmlu.mcf`
- `squad_it.default`
- `mkqa.entity`
- `mkqa.short_phrase`
- `mkqa.number`
- `mkqa.number_with_unit`
- `mkqa.date`
- `mkqa.binary`
- `mintaka.default`

## Recommended Repair Order

1. Keep `mkqa.long_answer` explicitly unsupported for the current bounded probe workflow unless a probe with actual Italian long-answer documents proves otherwise.
2. Use `configs/lighteval_verified_windows.yaml` or the `verified_windows` suite as the supported bounded LightEval path on local Windows.
3. Treat any future regressions in `mlmm_hellaswag.*`, `mlmm_arc_challenge.*`, or `squad_it.default` against the current wrapper as re-verification work, not as already-known unresolved blockers.

## Framework stabilization after the task sweep

- 2026-08-14: corrected overlapping-window perplexity accounting; this does not change LightEval task classifications.
- 2026-08-14: corrected checkpoint comparison keys for normalized LightEval metrics; task classifications are unchanged.
- 2026-08-14: pinned the tiny model/tokenizer in supported LightEval probe configs and added general reproducibility warnings; task classifications are unchanged.
- 2026-08-14: empty perplexity runs now fail explicitly instead of returning `1.0`; LightEval task classifications are unchanged.
- 2026-08-14: reconciled README documentation with the normalized LightEval output schema; task classifications are unchanged.
- 2026-08-14: added normalization coverage for expanded tasks, aggregates, few-shot settings, and legacy results; task classifications are unchanged.
- 2026-08-14: added evaluation-config discovery so support YAML is excluded from config audits; task classifications are unchanged.
- 2026-08-14: extended the normalized metric-row schema to every evaluation component; LightEval task classifications are unchanged.
- 2026-08-14: added non-blocking compatibility warnings for result comparisons; task classifications are unchanged.
- 2026-08-14: LightEval results now include the Hugging Face dataset revision inventory from the isolated cache; task classifications are unchanged.
- 2026-08-14: added network-free Python 3.12 CI for compilation and unit tests; task classifications are unchanged.
- 2026-08-14: re-ran the complete bounded non-LightEval suite successfully after the reporting changes; task classifications are unchanged.
- 2026-08-14: finalized the CI test dependency installation; task classifications are unchanged.
- 2026-08-14: fixed Linux CI assertions for Windows path compatibility helpers; task classifications are unchanged.
- 2026-08-14: rebuilt the real-model Colab launcher with bounded defaults and optional `verified_windows` LightEval; task classifications are unchanged.
- 2026-08-14: added the supported public Python evaluation entry point; task classifications are unchanged.
- 2026-08-14: added the unified `it-eval evaluate` command; task classifications are unchanged.
- 2026-08-14: added package-owned presets, including the bounded supported `verified_windows` LightEval suite; task classifications are unchanged.
- 2026-08-14: documented the separate pinned LightEval installation required by external-project users; task classifications are unchanged.
- 2026-08-14: added PyPI-ready package metadata and a tag release workflow; LightEval remains an explicitly managed optional runtime and task classifications are unchanged.
- 2026-08-14: validated the external wheel and public quick-preset API end to end; LightEval task classifications are unchanged.
- 2026-08-14: adopted Apache-2.0 package licensing for the release; LightEval task classifications are unchanged.
