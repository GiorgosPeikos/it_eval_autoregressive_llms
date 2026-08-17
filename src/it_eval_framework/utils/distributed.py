from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as dist

from it_eval_framework.config import ModelConfig


T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def owns(self, index: int) -> bool:
        return index % self.world_size == self.rank

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def gather(self, value: T) -> list[T] | None:
        if not self.enabled:
            return [value]
        gathered: list[T | None] | None = [None] * self.world_size if self.is_main else None
        dist.gather_object(value, gathered, dst=0)
        return gathered if self.is_main else None  # type: ignore[return-value]

    def broadcast(self, value: T | None) -> T:
        if not self.enabled:
            return value  # type: ignore[return-value]
        values = [value]
        dist.broadcast_object_list(values, src=0)
        return values[0]  # type: ignore[return-value]


_CONTEXT: DistributedContext | None = None


def initialize_distributed(parallelism: str = "auto", configured_device: str = "cpu") -> DistributedContext:
    """Initialize a torchrun process group for replicated inference."""
    global _CONTEXT
    if _CONTEXT is not None:
        return _CONTEXT

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    launched_distributed = world_size > 1

    if launched_distributed and parallelism in {"single", "model_parallel"}:
        raise RuntimeError(
            f"torchrun started {world_size} processes but runtime.parallelism={parallelism!r}. "
            "Use data_parallel/auto, or launch model_parallel with one process."
        )

    enabled = launched_distributed and parallelism in {"auto", "data_parallel"}
    device = configured_device
    if enabled:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
            backend = "nccl"
        else:
            device = "cpu"
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
    elif configured_device == "cuda" and torch.cuda.is_available():
        device = "cuda:0"

    _CONTEXT = DistributedContext(enabled, rank, world_size, local_rank, device)
    return _CONTEXT


def flatten_gathered(gathered: list[list[T]] | None) -> list[T]:
    if gathered is None:
        return []
    return [item for partition in gathered for item in partition]


def local_model_config(model_config: ModelConfig, context: DistributedContext) -> ModelConfig:
    if not context.enabled:
        return model_config
    return model_config.model_copy(update={"device": context.device, "device_map": None})


def generation_seed(base_seed: int, prompt_id: str, profile_name: str) -> int:
    payload = f"{base_seed}:{prompt_id}:{profile_name}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31)
