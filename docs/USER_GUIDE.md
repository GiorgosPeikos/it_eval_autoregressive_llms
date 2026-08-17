# Evaluate your Italian language model

The framework accepts decoder-only Hugging Face-compatible base models from either:

- a Hugging Face repository id, such as `my-org/my-italian-model`
- a local `save_pretrained()` checkpoint directory

Start with `quick`. It is deliberately bounded and checks BLiMP-IT, Italian perplexity, generation, aggregation, and reproducibility output. Its scores are not publication measurements.

## Route A: Colab

Open `notebooks/colab_model_eval_template.ipynb` with the README badge, select a GPU runtime, and edit the first settings cell.

For a Hub model:

```python
MODEL_SOURCE = "my-org/my-italian-model"
MODEL_REVISION = None  # Automatically resolved to the current immutable commit SHA.
```

Model and tokenizer revisions are resolved independently. If both sources name the
same Hub repository, the resolved SHA is shared. If `tokenizer_source` names a
different repository, omit `tokenizer_revision` to resolve that repository's own
current SHA. The resolved values are saved in `resolved_config.yaml` and
`run_config.yaml` inside the run directory. Supply explicit commit SHAs when you
need to reproduce an older snapshot rather than the current repository state.

For a checkpoint stored in Google Drive:

```python
MOUNT_GOOGLE_DRIVE = True
MODEL_SOURCE = "/content/drive/MyDrive/models/my-checkpoint"
```

Run the bounded defaults first. Download the ZIP from the final cell and retain it as the run record.

## Route B: install into another project

Install the released package from PyPI:

```bash
python -m pip install it-eval-framework
```

To test an unreleased commit instead, install directly from GitHub:

```bash
python -m pip install "it-eval-framework @ git+https://github.com/GiorgosPeikos/it_eval_autoregressive_llms.git"
```

## Maintainer release checklist

The repository builds as the `it-eval-framework` distribution.

The `it-eval-framework` PyPI project and trusted publisher are configured. For later releases:

1. update and test the version in `pyproject.toml`
2. commit and push the release state
3. push the matching version tag; the workflow builds, validates, and publishes the distributions

The first trusted-publishing release, `v0.1.0`, completed successfully on 2026-08-14. Version `0.1.1` followed as a documentation-only patch. Version `0.1.2` adds the complete evaluation walkthroughs, hardened LightEval execution, automatic revision handling, bounded streaming perplexity, adaptive multi-GPU evaluation, and explained result reporting.

Then evaluate a Hub model:

```bash
it-eval evaluate \
  --model my-org/my-italian-model \
  --revision IMMUTABLE_COMMIT \
  --preset quick \
  --device auto
```

Or a local checkpoint:

```bash
it-eval evaluate \
  --model ./checkpoints/model-10000 \
  --tokenizer ./tokenizer \
  --artifact-sha256 YOUR_ARCHIVE_DIGEST \
  --preset quick \
  --device cuda
```

From Python:

```python
from it_eval_framework import evaluate

run_dir = evaluate(
    model="my-org/my-italian-model",
    revision="IMMUTABLE_COMMIT",
    preset="quick",
    device="auto",
)
print(run_dir)
```

## Presets

| Preset | Purpose | Bounded? | LightEval |
|---|---|---:|---:|
| `quick` | First end-to-end integration check | yes | no |
| `perplexity` | Fast Italian corpus/model compatibility check | yes | no |
| `verified_windows` | Supported bounded LightEval task set | yes | yes |
| `full` | All offered evaluable tasks/corpora/prompts | no | yes (`all`, 39 variants) |

Do not begin with `full`. First prove that `quick` works, then choose the components and limits appropriate to the research question.

The LightEval component has its own suite selector. `quick` selects 1 task, `full`
selects the 33-task curated suite, `verified_windows` selects 39 task variants that
have been retained in the broad verified set, and `all` selects all 39 currently
evaluable Italian task variants. The known `mkqa.long_answer` alias is excluded
because it produces zero documents with the pinned runtime. `all` is exhaustive
with respect to this repository's evaluable Italian registry, not every language
and task distributed by LightEval. See [Running LightEval reliably](LIGHTEVAL_README.md)
before starting a broad run.

`lighteval.max_samples: null` means every available example in each selected task.
It has no effect while `lighteval.enabled` is false. Likewise,
`blimp_it.max_samples: null` means every example across every selected BLiMP-IT
subset.

### Controlling remote perplexity corpus size

Remote perplexity datasets stream by default. Streaming avoids materializing unused
splits and lets `max_documents` stop both download and evaluation work early. Use a
fixed document budget for portable, reproducible comparisons; set it to `null` only
when you intend to score the complete configured split.

```yaml
perplexity:
  dataset_streaming: true
  max_documents: 1000       # null means the complete selected split
  max_tokens_per_document: 2048
```

Dataset subsets are repository-specific. For `gsarti/clean_mc4_it`, the validation
shards represented by `tiny`, `small`, `medium`, `large`, and `full` are respectively
1/8, 1/4, 1/2, 3/4, and the complete validation corpus. A fixed `max_documents` budget
is generally preferable to a fraction because it controls runtime across datasets
without first scanning the complete corpus.

## Installing LightEval support

The pinned Italian LightEval path requires a resolver-managed installation because LightEval's declared `datasets` requirement conflicts with task scripts that still require `datasets==3.6.0`.

```bash
python -m pip install --upgrade "pip<27" "setuptools<82" wheel
python -m pip install "lighteval[multilingual]==0.13.0" --no-deps
python -m pip install -r https://raw.githubusercontent.com/GiorgosPeikos/it_eval_autoregressive_llms/main/constraints/lighteval-python310-313.txt
python -m pip install "it-eval-framework==0.1.2" --no-deps
```

Use Python 3.10–3.13. Authenticate with `hf auth login` before a large task sweep.

## Results

For formulas and interpretation—including what accuracy `1.0` means, how BLiMP-IT
and LightEval aggregate correctness, when perplexity is comparable, and how to read
generation diagnostics—see [Understanding evaluation results](RESULTS_README.md).

The returned run directory is the evaluation artifact. Preserve the complete directory, including:

- `run_config.yaml`
- `environment.json`
- `reproducibility.json`
- component result files and raw logs
- `summary.csv`
- `report.md`

If `reproducibility.json` reports `fully_pinned_inputs: false`, resolve its listed issues before treating the run as archival.

## Advanced configuration

Use YAML only when presets are insufficient:

```bash
it-eval evaluate --config my_model_eval.yaml
```

Copy `configs/local_model_example.yaml` when working inside a repository clone. The component-specific `it-eval-run-*` commands remain available for debugging.
