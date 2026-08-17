"""Two-process smoke worker invoked by test_distributed.py."""

from it_eval_framework.utils.distributed import initialize_distributed


context = initialize_distributed("data_parallel", "cpu")
owned = [index for index in range(8) if context.owns(index)]
gathered = context.gather(owned)
if context.is_main:
    assert sorted(index for partition in gathered for index in partition) == list(range(8))
context.barrier()
