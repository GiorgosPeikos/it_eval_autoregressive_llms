from __future__ import annotations

LIGHTEVAL_VERSION = "0.13.0"

# Verified locally from the lighteval 0.13.0 multilingual registry on 2026-08-14.
ITALIAN_LIGHTEVAL_TASKS = {
    "mlmm_hellaswag": {
        "cf": "mlmm_hellaswag_ita_cf",
        "mcf": "mlmm_hellaswag_ita_mcf",
        "hybrid": "mlmm_hellaswag_ita_hybrid",
    },
    "xcopa": {
        "cf": "xcopa_ita_cf",
        "mcf": "xcopa_ita_mcf",
        "hybrid": "xcopa_ita_hybrid",
    },
    "xcsqa": {
        "cf": "xcsqa_ita_cf",
        "mcf": "xcsqa_ita_mcf",
        "hybrid": "xcsqa_ita_hybrid",
    },
    "xcodah": {
        "cf": "xcodah_ita_cf",
        "mcf": "xcodah_ita_mcf",
        "hybrid": "xcodah_ita_hybrid",
    },
    "global_mmlu": {
        "mcf": "global_mmlu_all_ita_mcf",
    },
    "mlmm_mmlu": {
        "cf": "mlmm_mmlu_ita_cf",
        "mcf": "mlmm_mmlu_ita_mcf",
        "hybrid": "mlmm_mmlu_ita_hybrid",
    },
    "mlmm_arc_challenge": {
        "cf": "mlmm_arc_ita_cf:challenge",
        "mcf": "mlmm_arc_ita_mcf:challenge",
        "hybrid": "mlmm_arc_ita_hybrid:challenge",
    },
    "m3exams": {
        "cf": "m3exams_ita_cf",
        "mcf": "m3exams_ita_mcf",
        "hybrid": "m3exams_ita_hybrid",
    },
    "exams": {
        "cf": "exams_ita_cf",
        "mcf": "exams_ita_mcf",
        "hybrid": "exams_ita_hybrid",
    },
    "mlmm_truthfulqa": {
        "cf_mc1": "mlmm_truthfulqa_ita_cf:mc1",
        "cf_mc2": "mlmm_truthfulqa_ita_cf:mc2",
        "mcf_mc1": "mlmm_truthfulqa_ita_mcf:mc1",
        "mcf_mc2": "mlmm_truthfulqa_ita_mcf:mc2",
        "hybrid_mc1": "mlmm_truthfulqa_ita_hybrid:mc1",
        "hybrid_mc2": "mlmm_truthfulqa_ita_hybrid:mc2",
    },
    "squad_it": {
        "default": "squad_ita",
    },
    "mkqa": {
        "entity": "mkqa_ita:entity",
        "short_phrase": "mkqa_ita:short_phrase",
        "long_answer": "mkqa_ita:long_answer",
        "number": "mkqa_ita:number",
        "number_with_unit": "mkqa_ita:number_with_unit",
        "date": "mkqa_ita:date",
        "binary": "mkqa_ita:binary",
    },
    "mintaka": {
        "default": "mintaka_ita",
    },
}

ALL_ITALIAN_LIGHTEVAL_ALIASES = [
    f"{group}.{variant}"
    for group, variants in ITALIAN_LIGHTEVAL_TASKS.items()
    for variant in variants
]

DEFAULT_LIGHTEVAL_SUITES = {
    "quick": [
        "squad_it.default",
    ],
    "full": [
        "mlmm_hellaswag.cf",
        "mlmm_hellaswag.mcf",
        "mlmm_hellaswag.hybrid",
        "xcopa.cf",
        "xcopa.mcf",
        "xcopa.hybrid",
        "global_mmlu.mcf",
        "mlmm_arc_challenge.cf",
        "mlmm_arc_challenge.mcf",
        "mlmm_arc_challenge.hybrid",
        "squad_it.default",
        "mkqa.entity",
        "mkqa.short_phrase",
        "mkqa.number",
        "mlmm_truthfulqa.cf_mc1",
        "mlmm_truthfulqa.cf_mc2",
        "mlmm_truthfulqa.mcf_mc1",
        "mintaka.default",
        "xcsqa.cf",
        "xcsqa.mcf",
        "xcsqa.hybrid",
        "xcodah.cf",
        "xcodah.mcf",
        "xcodah.hybrid",
        "m3exams.cf",
        "m3exams.mcf",
        "m3exams.hybrid",
        "exams.cf",
        "exams.mcf",
        "exams.hybrid",
        "mlmm_mmlu.cf",
        "mlmm_mmlu.mcf",
        "mlmm_mmlu.hybrid",
    ],
    "verified_windows": [
        "exams.cf",
        "exams.hybrid",
        "exams.mcf",
        "global_mmlu.mcf",
        "m3exams.cf",
        "m3exams.hybrid",
        "m3exams.mcf",
        "mintaka.default",
        "mkqa.binary",
        "mkqa.date",
        "mkqa.entity",
        "mkqa.number",
        "mkqa.number_with_unit",
        "mkqa.short_phrase",
        "mlmm_arc_challenge.cf",
        "mlmm_arc_challenge.hybrid",
        "mlmm_arc_challenge.mcf",
        "mlmm_hellaswag.cf",
        "mlmm_hellaswag.hybrid",
        "mlmm_hellaswag.mcf",
        "mlmm_mmlu.cf",
        "mlmm_mmlu.hybrid",
        "mlmm_mmlu.mcf",
        "mlmm_truthfulqa.cf_mc1",
        "mlmm_truthfulqa.cf_mc2",
        "mlmm_truthfulqa.hybrid_mc1",
        "mlmm_truthfulqa.hybrid_mc2",
        "mlmm_truthfulqa.mcf_mc1",
        "mlmm_truthfulqa.mcf_mc2",
        "squad_it.default",
        "xcodah.cf",
        "xcodah.hybrid",
        "xcodah.mcf",
        "xcopa.cf",
        "xcopa.hybrid",
        "xcopa.mcf",
        "xcsqa.cf",
        "xcsqa.hybrid",
        "xcsqa.mcf",
    ],
    # Every Italian task/variant registered by this framework. This includes
    # tasks omitted from the curated full/verified suites.
    "all": ALL_ITALIAN_LIGHTEVAL_ALIASES,
}


def resolve_task_alias(alias: str) -> str:
    group, variant = alias.split(".", 1)
    return ITALIAN_LIGHTEVAL_TASKS[group][variant]


def resolve_task_aliases(aliases: list[str]) -> list[str]:
    return [resolve_task_alias(alias) for alias in aliases]
