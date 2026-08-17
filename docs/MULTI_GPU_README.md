# Multi-GPU evaluation

This guide covers the two supported ways to use several GPUs. Choose the mode from the model's memory requirements; the modes solve different problems.

## 1. Replicated inference for higher throughput

Use replicated data parallelism when one complete model fits on one GPU. The framework starts one process per GPU, loads one model replica in each process, partitions evaluation work deterministically, and combines results on rank 0.

This is the recommended mode for Sophira-360M and other models that fit comfortably on a single device.

```yaml
model:
  source: Gpeik/Sophira-360M-base
  device: cuda
  dtype: auto

runtime:
  parallelism: data_parallel
  num_processes: auto
```

Launch using every visible GPU:

```bash
it-eval-launch --config configs/italian_base_quick.yaml
```

Limit the run to two GPUs either in YAML or on the command line:

```bash
CUDA_VISIBLE_DEVICES=0,1 it-eval-launch \
  --config configs/italian_base_quick.yaml \
  --num-processes 2
```

The equivalent low-level command is:

```bash
torchrun --standalone --nproc-per-node 2 \
  -m it_eval_framework.runners.run_all \
  --config configs/italian_base_quick.yaml
```

On Windows PowerShell, set visible devices before launching:

```powershell
$env:CUDA_VISIBLE_DEVICES = "0,1"
it-eval-launch --config configs/italian_base_quick.yaml --num-processes 2
```

`runtime.parallelism: auto` is a convenient default: `it-eval-launch` uses replicated inference when it sees multiple GPUs and ordinary single-process execution otherwise. Running `python -m it_eval_framework.runners.run_all` directly remains single-process unless it is started by `torchrun`.

### How work is divided

- BLiMP-IT examples are assigned by stable global example index.
- Perplexity documents are assigned by document index. NLL and token totals are reduced before perplexity is calculated.
- Generation prompt/profile jobs are assigned by stable job index. Each job has a deterministic derived seed, so generated outputs do not depend on the GPU count.
- Only rank 0 writes run state, sample files, normalized results, `summary.csv`, and `report.md`.

Each GPU needs enough memory for a full model replica. `model.batch_size` is per process where a backend supports batching, so total effective capacity grows with the process count.

## 2. Shard a model that does not fit on one GPU

Use model parallelism when the model is too large for one GPU:

```yaml
model:
  source: my-org/my-large-model
  device: auto
  device_map: auto
  dtype: bfloat16
  max_memory:
    0: 22GiB
    1: 22GiB

runtime:
  parallelism: model_parallel
  num_processes: 1
```

Then run:

```bash
it-eval-launch --config my-large-model.yaml
```

Transformers and Accelerate place model layers across the visible GPUs. This primarily increases usable model memory; it is not expected to scale throughput like replicated inference. Do not combine `device_map` with `data_parallel` in the same run.

## LightEval behavior

The repository-owned BLiMP-IT, perplexity, and generation stages use the distributed implementation described above. During a replicated run, the integrated LightEval stage is executed once by rank 0 to avoid nested process groups and duplicate result writers. LightEval's own Accelerate and vLLM backends can be run separately when distributed LightEval throughput is required.

For a standalone LightEval data-parallel run, follow the command syntax for the installed LightEval version. A typical Accelerate launch is:

```bash
accelerate launch --multi_gpu --num_processes 4 -m lighteval accelerate \
  "model_name=my-org/my-model,dtype=bfloat16" \
  "leaderboard|truthfulqa:mc|0|0"
```

Keep standalone LightEval outputs separate from a framework run directory unless importing them through a controlled aggregation workflow.

## Cluster and scheduler practice

The built-in launcher targets one multi-GPU machine. On Slurm or another multi-node scheduler, let the scheduler provide ranks and use `torchrun` with the appropriate node count, node rank, rendezvous address, and rendezvous port. All ranks must see the same repository, config, model, dataset cache, and output directory.

Start with a one-node equivalence check before scaling:

1. Run a bounded evaluation on one GPU.
2. Run the same pinned config on two GPUs.
3. Confirm aggregate BLiMP accuracy and perplexity match within normal floating-point tolerance.
4. Confirm the same generation rows and effective seeds are present.
5. Increase sample limits only after the equivalence check succeeds.

Avoid `torch.nn.DataParallel`. One process per GPU is the standard distributed execution model and avoids a single Python process becoming the input and output bottleneck.

Official references: [PyTorch distributed overview](https://docs.pytorch.org/docs/stable/distributed.html), [Transformers big-model inference](https://huggingface.co/docs/transformers/main/en/big_models), [Accelerate distributed inference](https://huggingface.co/docs/accelerate/main/en/usage_guides/distributed_inference), and [LightEval quick tour](https://huggingface.co/docs/lighteval/quicktour).

## Troubleshooting

- **Every GPU repeats all samples:** use `it-eval-launch` or `torchrun`; starting several unrelated Python commands does not create a process group.
- **CUDA out of memory immediately:** replicated mode requires a complete model on every GPU. Use `model_parallel` with `device_map: auto`, lower precision, or a smaller model.
- **The run waits at a barrier:** inspect every rank's first exception. `torchrun` terminates the group when one rank fails, but scheduler logs may separate rank output.
- **Wrong GPUs are selected:** set `CUDA_VISIBLE_DEVICES` before launching and keep `num_processes` at or below the number of visible devices.
- **Only one GPU is used:** check `torch.cuda.device_count()`, `runtime.parallelism`, and the launcher's printed process count.
