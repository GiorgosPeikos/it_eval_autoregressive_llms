# Understanding evaluation results

This guide explains what every result family means, how its aggregate is calculated, and what can and cannot be concluded from it. Always interpret a value together with its task, formulation, sample count, dataset split, and configuration.

## First principle: a score belongs to an evaluation specification

`accuracy = 1.0` does not mean “the model is perfect.” It means every item included in that particular metric calculation received a score of 1. If a smoke run contains two items, 1.0 means 2/2. If a complete dataset contains 1,000 items, 1.0 means 1,000/1,000. Those results have radically different evidential strength.

For every score, retain:

- component and task ID;
- metric name and formulation;
- sample count and few-shot count;
- model and tokenizer revisions;
- dataset revision, subset, and split;
- preprocessing, context length, stride, and token limits;
- repository and dependency versions.

## Normalized summary columns

| Column | Meaning |
|---|---|
| `component` | `lighteval`, `blimp_it`, `perplexity`, or `generation` |
| `task_id` | Exact benchmark, phenomenon, corpus, or aggregate identity |
| `fewshot` | Number of demonstrations used by the task |
| `metric` | Name of the calculation; never interpret `value` without it |
| `value` | Raw metric value, generally not converted to a percentage |
| `stderr` | LightEval-reported standard error when available |
| `sample_count` | Number of documents/examples or scored tokens used as the denominator |
| `higher_is_better` | Direction supplied by the task/runner when known |

The notebook adds `range`, `direction`, and `meaning` columns for readability. The raw value remains unchanged.

## LightEval metrics

LightEval provides task-specific prompts, inference requests, metrics, and corpus aggregation. This repository preserves the raw result JSON and converts its numeric metrics into the normalized table.

### Multiple-choice accuracy

For a task with evaluated examples indexed by `i`, define:

```text
correct_i = 1 if the model-selected choice is a gold choice, otherwise 0
accuracy = sum(correct_i) / N
```

The selected choice is the choice with the best score under that task formulation. Formulations are separate experiments:

- `cf`: conditional likelihood of the continuation under the context;
- `mcf`: LightEval's multiple-choice formulation;
- `hybrid`: LightEval's hybrid multiple-choice formulation.

Metric names such as `acc`, `accuracy`, `loglikelihood_acc`, or `acc_norm` are accuracy-like, but the exact score and normalization are defined by the task metadata. A length-normalized metric can choose a different answer from raw total log-likelihood. Do not average `cf`, `mcf`, and `hybrid` as if they were repeated measurements of an identical procedure.

For a balanced `k`-choice task, uniform random choice has expected accuracy `1/k`; use the actual task choice distribution before claiming a chance baseline.

### Exact match

```text
exact_i = 1 if normalized_prediction_i matches an accepted normalized_reference_i
exact_match = sum(exact_i) / N
```

An exact-match score of 0 can still contain semantically useful answers that differ in spelling or formatting. Conversely, exact match does not establish that an answer is well explained.

### F1

F1 is the harmonic mean of precision and recall under the task's overlap units, often answer tokens:

```text
F1 = 2 × precision × recall / (precision + recall)
```

Task configuration determines normalization and whether aggregation is sample mean, macro, or micro. Inspect the raw LightEval configuration before comparing differently named F1 metrics.

### Standard error

`stderr` estimates uncertainty of a sample aggregate. When a normal approximation is reasonable, an approximate 95% interval is:

```text
value ± 1.96 × stderr
```

This does not cover dataset contamination, translation artifacts, prompt sensitivity, model nondeterminism, or benchmark validity. Tiny bounded probes are pipeline tests, not reliable estimates.

### Subject-expanded tasks

Global MMLU, MLMM MMLU, and EXAMS can produce subject-level rows. A macro average weights subjects equally; a micro average weights examples equally. Preserve subject rows and state explicitly which aggregate you report.

## BLiMP-IT accuracy

BLiMP-IT contains Italian minimal pairs: a grammatical sentence `g_i` and a minimally different ungrammatical sentence `u_i`. The framework computes teacher-forced total sequence log-probability:

```text
score(x) = sum over next-token log P(token_t | tokens before t)
correct_i = 1 if score(g_i) >= score(u_i), otherwise 0
accuracy = sum(correct_i) / N
```

Overall accuracy uses all evaluated pairs. Each phenomenon accuracy uses only pairs labeled with that phenomenon:

```text
phenomenon_accuracy_p = correct pairs in p / evaluated pairs in p
```

`accuracy = 1.0` means the model preferred the grammatical member for every evaluated pair. It does not imply perfect grammar generation, and total sequence likelihood can be affected by tokenization and sentence length. Compare only runs using the same scoring rule and dataset revision.

## Held-out token perplexity

For all scored target tokens, the runner accumulates negative log-likelihood (NLL):

```text
mean_loss = total_NLL / total_scored_target_tokens
token_perplexity = exp(mean_loss)
```

Lower is better. Perplexity is not accuracy and is not bounded by 1 above; 1 is the theoretical lower limit corresponding to probability 1 on every observed target token.

A loose intuition is an effective average uncertainty over next-token choices, but it should not be interpreted as a literal vocabulary branching count. Perplexity values are comparable only when all of the following match:

- tokenizer;
- corpus revision, subset, and split;
- text normalization and special-token policy;
- sequence length and stride;
- document-boundary policy;
- document and token budgets.

A lower perplexity on text included in training is not held-out generalization. Use a corpus that is document-wise or temporally separated from training data whenever possible.

## Controlled generation diagnostics

The generation component runs every selected prompt under every decoding profile. It saves text and diagnostics per row; `num_generations` is only a coverage count.

| Diagnostic | Calculation | Interpretation |
|---|---|---|
| `output_length_chars/words` | Output size | Descriptive; truncation and verbosity both affect it |
| `distinct_1/2/3` | Unique n-grams divided by all output n-grams | Higher indicates more local diversity, not necessarily higher quality |
| `repeated_3gram_rate` | Repeated trigram occurrences divided by trigram count | Lower usually indicates less repetitive degeneration |
| `unfinished_output` | Final-character punctuation heuristic | A review flag, not a semantic judgment |
| `artifact_flags` | Encoding and long-repeat detectors | Potential corruption/degeneration symptoms |
| `generation_latency_seconds` | Wall-clock generation time | Hardware- and configuration-dependent |

Human evaluation is still required for fluency, coherence, instruction relevance, truthfulness, toxicity, bias, and stylistic quality. Compare greedy and sampled profiles separately.

## Result files

| File | Purpose |
|---|---|
| `resolved_config.yaml` | Immutable revisions and expanded task aliases actually used |
| `run_config.yaml` | Validated component configuration |
| `environment.json` | Python/package/hardware and repository snapshot |
| `reproducibility.json` | Pinning audit and config identity |
| `benchmark_results.json` | Normalized and raw LightEval results |
| `lighteval_raw/` | Original LightEval files and optional details |
| `blimp_it_results.json` | Overall and phenomenon aggregates |
| `blimp_it_samples.jsonl` | Pair-level scores and correctness |
| `perplexity_results.json` | Token totals, loss, perplexity, and optional document stats |
| `generations.jsonl` | Prompt/profile text and diagnostics |
| `summary.csv` | Cross-component normalized metric rows |
| `report.md` | Markdown rendering of the normalized summary |
| `run_state.json` | Completed, running, or failed stage state |

## Comparing checkpoints

Only compare rows with identical component, task ID, metric, few-shot setting, and evaluation configuration. A score difference is not meaningful when one run changed the tokenizer, sample cap, corpus, prompt formulation, dataset revision, or decoding profile.

For research reporting, include both aggregate scores and enough per-task/per-phenomenon detail to reveal uneven behavior. Never replace the original benchmark citation with only a citation to this framework.
