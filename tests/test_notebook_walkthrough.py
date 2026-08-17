import json
from pathlib import Path


NOTEBOOK = Path("notebooks/colab_model_eval_template.ipynb")


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def test_model_eval_notebook_is_a_complete_bounded_walkthrough():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(_source(cell) for cell in notebook["cells"])

    assert 'EVALUATION_PROFILE = "smoke"' in source
    assert "ALLOW_UNBOUNDED_FULL_RUN = False" in source
    assert 'lighteval_suite="all"' in source
    assert 'ppl_subset="full"' in source
    assert "ENABLE_LIGHTEVAL = True" in source
    assert "ENABLE_BLIMP_IT = True" in source
    assert "ENABLE_PERPLEXITY = True" in source
    assert "ENABLE_GENERATION = True" in source
    assert "annotate_metric_rows" in source
    assert "accuracy `1.0`" in source


def test_model_eval_notebook_python_cells_compile_after_removing_colab_magics():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        python_lines = [
            line
            for line in _source(cell).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        compile("\n".join(python_lines), f"{NOTEBOOK}:cell-{index}", "exec")
