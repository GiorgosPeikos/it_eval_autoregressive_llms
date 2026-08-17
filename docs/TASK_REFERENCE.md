# Task Reference

This document is the paper-facing reference for the tasks used in this repository.

It answers four questions:

1. what each task measures
2. how it is formulated in this repository
3. which aliases and task ids are used
4. what a user should report in a paper for reproducibility

## Reporting checklist

If you report results from this repository, include at least:

- repository commit or release
- model name and checkpoint
- `lighteval==0.13.0`
- `datasets==3.6.0`
- the exact config file used
- the exact task aliases or suite name
- whether the run used the bounded `verified_windows` subset or another suite
- `max_samples` if you used a bounded smoke or probe run
- device, dtype, and batch size

For LightEval-based tasks, also report the formulation:

- `cf`: conditional log-likelihood
- `mcf`: multiple-choice formulation as exposed by the task
- `hybrid`: LightEval hybrid multiple-choice formulation

## Repository evaluation families

The repository currently contains four evaluation families:

1. LightEval-backed benchmark tasks
2. BLiMP-IT minimal-pair evaluation
3. held-out perplexity
4. controlled generation

Only the first family is task-aliased through `src/it_eval_framework/task_registry.py`.

## Stable user-facing paths

Current user-facing paths are:

- `configs/italian_base_quick.yaml`
  - smoke test
  - LightEval disabled
  - BLiMP-IT, Italian perplexity, and generation enabled
- `configs/lighteval_verified_windows.yaml`
  - supported bounded LightEval subset on the validated local Windows path
- `configs/local_model_example.yaml`
  - starting point for user model evaluation

## LightEval task groups

### XCOPA

- Aliases:
  - `xcopa.cf`
  - `xcopa.mcf`
  - `xcopa.hybrid`
- Resolved task ids:
  - `xcopa_ita_cf`
  - `xcopa_ita_mcf`
  - `xcopa_ita_hybrid`
- What it measures:
  - causal commonsense reasoning
  - the model chooses the more plausible cause or effect continuation
- Task type:
  - multiple-choice commonsense reasoning
- Source benchmark:
  - XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning
  - https://aclanthology.org/2020.emnlp-main.185/
- Notes:
  - Italian is a translated language variant inside the multilingual XCOPA benchmark

### XCSQA

- Aliases:
  - `xcsqa.cf`
  - `xcsqa.mcf`
  - `xcsqa.hybrid`
- Resolved task ids:
  - `xcsqa_ita_cf`
  - `xcsqa_ita_mcf`
  - `xcsqa_ita_hybrid`
- What it measures:
  - broad commonsense question answering
- Task type:
  - 5-way multiple-choice QA
- Source benchmark:
  - CommonsenseQA
  - https://aclanthology.org/N19-1421/
  - Cross-lingual benchmark context:
  - https://inklab.usc.edu/XCSR/xcsr_datasets
  - https://arxiv.org/abs/2106.06937
- Notes:
  - the Italian task in this repo is the X-CSR Italian cross-lingual variant of CommonsenseQA

### XCODAH

- Aliases:
  - `xcodah.cf`
  - `xcodah.mcf`
  - `xcodah.hybrid`
- Resolved task ids:
  - `xcodah_ita_cf`
  - `xcodah_ita_mcf`
  - `xcodah_ita_hybrid`
- What it measures:
  - adversarial commonsense sentence completion
- Task type:
  - multiple-choice sentence completion
- Source benchmark:
  - CODAH: An Adversarially-Authored Question Answering Dataset for Common Sense
  - https://aclanthology.org/W19-2008/
  - Cross-lingual benchmark context:
  - https://inklab.usc.edu/XCSR/xcsr_datasets
  - https://arxiv.org/abs/2106.06937
- Notes:
  - the Italian task in this repo is the X-CSR Italian cross-lingual variant of CODAH

### MLMM HellaSwag

- Aliases:
  - `mlmm_hellaswag.cf`
  - `mlmm_hellaswag.mcf`
  - `mlmm_hellaswag.hybrid`
- Resolved task ids:
  - `mlmm_hellaswag_ita_cf`
  - `mlmm_hellaswag_ita_mcf`
  - `mlmm_hellaswag_ita_hybrid`
- What it measures:
  - grounded commonsense inference over plausible next-event continuations
- Task type:
  - multiple-choice continuation selection
- Source benchmark:
  - HellaSwag: Can a Machine Really Finish Your Sentence?
  - https://aclanthology.org/P19-1472/
- Notes:
  - the Italian task is exposed through the multilingual LightEval task registry via `okapi_hellaswag`

### Global MMLU

- Aliases:
  - `global_mmlu.mcf`
- Resolved task id prefix:
  - `global_mmlu_all_ita_mcf`
- What it measures:
  - broad academic and professional knowledge in Italian
  - attention to multilingual and cultural evaluation quality
- Task type:
  - subject-expanded multiple-choice QA
- Source benchmark:
  - Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation
  - https://aclanthology.org/2025.acl-long.919.pdf
- Notes:
  - the task expands into subject-level evaluations under LightEval
  - report subject aggregation choices explicitly

### MLMM MMLU

- Aliases:
  - `mlmm_mmlu.cf`
  - `mlmm_mmlu.mcf`
  - `mlmm_mmlu.hybrid`
- Resolved task ids:
  - `mlmm_mmlu_ita_cf`
  - `mlmm_mmlu_ita_mcf`
  - `mlmm_mmlu_ita_hybrid`
- What it measures:
  - broad multitask knowledge across MMLU subject areas
- Task type:
  - subject-expanded multiple-choice QA
- Source benchmark:
  - Measuring Massive Multitask Language Understanding
  - https://openreview.net/forum?id=d7KBjmI3GmQ
- Notes:
  - the task expands into many subjects under LightEval
  - for papers, specify whether you report macro averages, raw subject scores, or both

### MLMM ARC Challenge

- Aliases:
  - `mlmm_arc_challenge.cf`
  - `mlmm_arc_challenge.mcf`
  - `mlmm_arc_challenge.hybrid`
- Resolved task ids:
  - `mlmm_arc_ita_cf:challenge`
  - `mlmm_arc_ita_mcf:challenge`
  - `mlmm_arc_ita_hybrid:challenge`
- What it measures:
  - grade-school science reasoning
- Task type:
  - multiple-choice science QA
- Source benchmark:
  - Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
  - https://arxiv.org/abs/1803.05457
- Notes:
  - this repository uses the `challenge` split, not the easy split

### M3ExamS

- Aliases:
  - `m3exams.cf`
  - `m3exams.mcf`
  - `m3exams.hybrid`
- Resolved task ids:
  - `m3exams_ita_cf`
  - `m3exams_ita_mcf`
  - `m3exams_ita_hybrid`
- What it measures:
  - exam-style knowledge and reasoning from real educational questions
- Task type:
  - multiple-choice exam QA
- Source benchmark:
  - M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models
  - https://arxiv.org/abs/2306.05179
- Notes:
  - the LightEval task used here is text-only at evaluation time in this repo
  - the original benchmark is multimodal and multilevel, which should be stated in any paper discussion

### Exams

- Aliases:
  - `exams.cf`
  - `exams.mcf`
  - `exams.hybrid`
- Resolved task ids:
  - `exams_ita_cf`
  - `exams_ita_mcf`
  - `exams_ita_hybrid`
- What it measures:
  - multilingual high-school examination question answering
- Task type:
  - subject-expanded multiple-choice QA
- Source benchmark:
  - EXAMS: A Multi-Subject High School Examinations Dataset for Cross-Lingual and Multilingual Question Answering
  - https://huggingface.co/papers/2011.03080
  - Dataset card:
  - https://huggingface.co/datasets/mhardalov/exams
- Notes:
  - exam questions cover multiple school subjects and languages
  - LightEval expands the evaluation by subject

### TruthfulQA

- Aliases:
  - `mlmm_truthfulqa.cf_mc1`
  - `mlmm_truthfulqa.cf_mc2`
  - `mlmm_truthfulqa.mcf_mc1`
  - `mlmm_truthfulqa.mcf_mc2`
  - `mlmm_truthfulqa.hybrid_mc1`
  - `mlmm_truthfulqa.hybrid_mc2`
- Resolved task ids:
  - `mlmm_truthfulqa_ita_cf:mc1`
  - `mlmm_truthfulqa_ita_cf:mc2`
  - `mlmm_truthfulqa_ita_mcf:mc1`
  - `mlmm_truthfulqa_ita_mcf:mc2`
  - `mlmm_truthfulqa_ita_hybrid:mc1`
  - `mlmm_truthfulqa_ita_hybrid:mc2`
- What it measures:
  - truthfulness under common misconceptions and false beliefs
- Task type:
  - multiple-choice truthfulness evaluation
- Source benchmark:
  - TruthfulQA: Measuring How Models Mimic Human Falsehoods
  - https://aclanthology.org/2022.acl-long.229/
- Notes:
  - `mc1` and `mc2` are distinct evaluation variants and should be reported separately

### SQuAD-it

- Aliases:
  - `squad_it.default`
- Resolved task id:
  - `squad_ita`
- What it measures:
  - Italian reading comprehension / answer generation from context
- Task type:
  - generative QA in the current LightEval path
- Source benchmark:
  - Dataset card:
  - https://huggingface.co/datasets/crux82/squad_it
  - Project repository:
  - https://github.com/crux82/squad-it
  - Original English benchmark:
  - SQuAD: 100,000+ Questions for Machine Comprehension of Text
  - https://arxiv.org/abs/1606.05250
- Notes:
  - SQuAD-it is a semi-automatic Italian translation/adaptation of SQuAD

### MKQA

- Aliases:
  - `mkqa.entity`
  - `mkqa.short_phrase`
  - `mkqa.long_answer`
  - `mkqa.number`
  - `mkqa.number_with_unit`
  - `mkqa.date`
  - `mkqa.binary`
- Resolved task id prefix:
  - `mkqa_ita:*`
- What it measures:
  - multilingual open-domain question answering across answer types
- Task type:
  - generative QA grouped by answer class
- Source benchmark:
  - MKQA: A Linguistically Diverse Benchmark for Multilingual Open Domain Question Answering
  - https://arxiv.org/abs/2007.15207
- Notes:
  - this repository evaluates separate answer-type subsets rather than one merged score
  - `mkqa.long_answer` is currently excluded from the supported bounded probe path because the pinned snapshot yields no evaluable documents for the Italian long-answer slice

### Mintaka

- Aliases:
  - `mintaka.default`
- Resolved task id:
  - `mintaka_ita`
- What it measures:
  - multilingual complex question answering with entity-centric answers
- Task type:
  - generative QA
- Source benchmark:
  - Mintaka: A Complex, Natural, and Multilingual Dataset for End-to-End Question Answering
  - https://aclanthology.org/2022.coling-1.138/
- Notes:
  - Mintaka contains complex question phenomena such as superlatives, intersections, and multi-hop reasoning

## Non-LightEval evaluation components

### BLiMP-IT

- Config block:
  - `blimp_it`
- What it measures:
  - grammatical acceptability preferences for minimal sentence pairs in Italian
- Task type:
  - pairwise sentence likelihood comparison
- Metric used in this repository:
  - pairwise accuracy
- Source dataset:
  - Dataset card:
  - https://huggingface.co/datasets/NeTSlab/BLiMP-IT
  - Original BLiMP benchmark:
  - BLiMP: The Benchmark of Linguistic Minimal Pairs for English
  - https://aclanthology.org/2020.tacl-1.25/
- Notes:
  - this repo uses direct base-model likelihood comparison between the grammatical and ungrammatical member of each pair

### Held-out perplexity

- Config block:
  - `perplexity`
- What it measures:
  - next-token predictive fit on held-out text
- Evaluation language in this repository:
  - Italian only
- Task type:
  - sliding-window negative log-likelihood
- Metric used in this repository:
  - token perplexity
- Notes:
  - the repository example uses `gsarti/clean_mc4_it` with `dataset_subset: tiny` and `split: validation`
  - for publication-grade evaluation, users should replace that example with their own genuinely held-out Italian corpus when possible
  - perplexity is only comparable when tokenization, preprocessing, and dataset separation are controlled

### Controlled generation

- Config block:
  - `generation`
- What it measures:
  - open-ended continuation behavior under fixed prompt sets and decoding profiles
- Task type:
  - free generation
- Output used in this repository:
  - generated continuations and diagnostics, not a single benchmark score
- Notes:
  - generation outputs are useful for qualitative analysis and failure inspection
  - they should not be presented as a substitute for benchmark accuracy or held-out perplexity

## Supported bounded LightEval subset

The currently supported local Windows LightEval subset is the `verified_windows` suite:

The `all` suite selects the same 39 currently evaluable Italian task variants on
every supported platform. The registry also knows `mkqa.long_answer`, but that
variant is intentionally excluded from executable sweeps because it produces zero
evaluation documents with the pinned LightEval 0.13.0 stack. Installation,
preflight, Colab settings, and staged-run instructions are documented in
[Running LightEval reliably](LIGHTEVAL_README.md).

- `exams.cf`
- `exams.hybrid`
- `exams.mcf`
- `global_mmlu.mcf`
- `m3exams.cf`
- `m3exams.hybrid`
- `m3exams.mcf`
- `mintaka.default`
- `mkqa.binary`
- `mkqa.date`
- `mkqa.entity`
- `mkqa.number`
- `mkqa.number_with_unit`
- `mkqa.short_phrase`
- `mlmm_arc_challenge.cf`
- `mlmm_arc_challenge.hybrid`
- `mlmm_arc_challenge.mcf`
- `mlmm_hellaswag.cf`
- `mlmm_hellaswag.hybrid`
- `mlmm_hellaswag.mcf`
- `mlmm_mmlu.cf`
- `mlmm_mmlu.hybrid`
- `mlmm_mmlu.mcf`
- `mlmm_truthfulqa.cf_mc1`
- `mlmm_truthfulqa.cf_mc2`
- `mlmm_truthfulqa.hybrid_mc1`
- `mlmm_truthfulqa.hybrid_mc2`
- `mlmm_truthfulqa.mcf_mc1`
- `mlmm_truthfulqa.mcf_mc2`
- `squad_it.default`
- `xcodah.cf`
- `xcodah.hybrid`
- `xcodah.mcf`
- `xcopa.cf`
- `xcopa.hybrid`
- `xcopa.mcf`
- `xcsqa.cf`
- `xcsqa.hybrid`
- `xcsqa.mcf`

Excluded from that supported bounded subset:

- `mkqa.long_answer`

## Reproducibility note

The task definitions come from a mixture of:

- original English benchmarks
- multilingual benchmark releases
- Italian task registrations exposed through LightEval's multilingual registry

For publication, do not cite only this repository. Cite both:

1. this repository for the execution setup
2. the original benchmark paper or dataset card for the task itself
