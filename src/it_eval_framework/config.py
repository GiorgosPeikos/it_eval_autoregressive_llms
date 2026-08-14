from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from it_eval_framework.task_registry import DEFAULT_LIGHTEVAL_SUITES


class ModelConfig(BaseModel):
    source: str = Field(description="Local HF checkpoint path or Hugging Face repo id.")
    revision: str | None = None
    tokenizer_source: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    tokenizer_use_fast: bool = True
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    device: str = "cpu"
    batch_size: int = 1
    max_model_length: int | None = None
    artifact_sha256: str | None = Field(default=None, description="User-supplied digest for a local model artifact.")

    @property
    def tokenizer_id(self) -> str:
        return self.tokenizer_source or self.source


class OutputConfig(BaseModel):
    root_dir: str = "evaluation_results"
    overwrite: bool = False
    save_details: bool = True


class LightEvalConfig(BaseModel):
    enabled: bool = True
    suite: str | None = "quick"
    task_aliases: list[str] = Field(default_factory=list)
    num_fewshot_seeds: int = 1
    max_samples: int | None = None
    dataset_loading_processes: int = 1
    extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fill_suite(self) -> "LightEvalConfig":
        if self.suite:
            self.task_aliases = list(dict.fromkeys([*self.task_aliases, *DEFAULT_LIGHTEVAL_SUITES[self.suite]]))
        return self


class BliMPITConfig(BaseModel):
    enabled: bool = True
    dataset_repo: str = "NeTSlab/BLiMP-IT"
    dataset_subset: str | None = None
    dataset_revision: str | None = None
    dataset_trust_remote_code: bool = False
    split: str = "test"
    max_samples: int | None = None
    phenomenon_field: str = "field"
    good_field: str = "sentence_good"
    bad_field: str = "sentence_bad"
    text_fallback_fields: list[str] = Field(default_factory=lambda: ["good_sentence", "bad_sentence", "sentence_acceptable", "sentence_unacceptable"])


class PerplexityConfig(BaseModel):
    enabled: bool = True
    dataset_path: str | None = None
    dataset_repo: str | None = None
    dataset_subset: str | None = None
    dataset_revision: str | None = None
    dataset_trust_remote_code: bool = False
    dataset_streaming: bool = False
    split: str | None = None
    text_field: str = "text"
    sequence_length: int = 1024
    stride: int = 512
    preserve_document_boundaries: bool = True
    add_bos_token: bool = True
    add_eos_token: bool = False
    per_document_stats: bool = True
    max_documents: int | None = None
    max_tokens_per_document: int | None = None
    contamination_warning: str = (
        "Held-out perplexity is only comparable when the evaluation corpus is document-wise or temporally "
        "separated from training data."
    )

    @model_validator(mode="after")
    def require_dataset(self) -> "PerplexityConfig":
        if self.enabled and not (self.dataset_path or self.dataset_repo):
            raise ValueError("Perplexity evaluation requires either dataset_path or dataset_repo.")
        if self.dataset_path and self.dataset_repo:
            raise ValueError("Set only one of perplexity.dataset_path or perplexity.dataset_repo.")
        return self


class DecodingProfile(BaseModel):
    name: str
    do_sample: bool
    temperature: float | None = None
    top_p: float | None = None
    max_new_tokens: int = 128


class GenerationPrompt(BaseModel):
    prompt_id: str
    category: str
    prompt_text: str


class GenerationConfig(BaseModel):
    enabled: bool = True
    prompts_path: str = "configs/generation_prompts.yaml"
    max_prompts: int | None = None
    seed: int = 13
    profiles: list[DecodingProfile] = Field(
        default_factory=lambda: [
            DecodingProfile(name="greedy", do_sample=False, max_new_tokens=128),
            DecodingProfile(name="temp_0_7_top_p_0_9", do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=128),
            DecodingProfile(name="temp_0_8_top_p_0_95", do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=128),
        ]
    )


class RuntimeConfig(BaseModel):
    seed: int = 13
    python_executable: str = "python"
    lighteval_command: str = "lighteval"


class EvaluationConfig(BaseModel):
    run_name: str = "italian_base_eval"
    model: ModelConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    lighteval: LightEvalConfig = Field(default_factory=LightEvalConfig)
    blimp_it: BliMPITConfig = Field(default_factory=BliMPITConfig)
    perplexity: PerplexityConfig | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: str | Path) -> EvaluationConfig:
    return EvaluationConfig.model_validate(load_yaml(path))
