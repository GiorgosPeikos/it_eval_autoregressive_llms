import json
from pathlib import Path


NOTEBOOK = Path("notebooks/colab_model_eval_template.ipynb")
QUICKSTART_NOTEBOOK = Path("notebooks/colab_quickstart.ipynb")


def _source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def test_model_eval_notebook_is_a_complete_bounded_walkthrough():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    cell_ids = {cell.get("metadata", {}).get("id") for cell in notebook["cells"]}

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
    assert {
        "lighteval-explanation",
        "lighteval-results",
        "blimp-explanation",
        "blimp-results",
        "perplexity-explanation",
        "perplexity-results",
        "generation-explanation",
        "generation-results",
        "overall-assessment-explanation",
        "overall-assessment",
    } <= cell_ids
    assert "There is intentionally no single combined score" in source


def test_model_eval_notebook_python_cells_compile_after_removing_colab_magics():
    for notebook_path in (NOTEBOOK, QUICKSTART_NOTEBOOK):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            python_lines = [
                line
                for line in _source(cell).splitlines()
                if not line.lstrip().startswith(("%", "!"))
            ]
            compile("\n".join(python_lines), f"{notebook_path}:cell-{index}", "exec")


def test_quickstart_matches_the_explanatory_result_structure():
    notebook = json.loads(QUICKSTART_NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(_source(cell) for cell in notebook["cells"])
    cell_ids = [cell.get("metadata", {}).get("id") for cell in notebook["cells"]]

    assert "ENABLE_LIGHTEVAL = False" in source
    assert "These are integration checks, not publication measurements" in source
    assert "annotate_metric_rows" in source
    assert "There is no scientifically valid universal total" in source
    for explanation_id, result_id in (
        ("lighteval-explanation", "lighteval-results"),
        ("blimp-explanation", "blimp-results"),
        ("perplexity-explanation", "perplexity-results"),
        ("generation-explanation", "generation-results"),
        ("overall-assessment-explanation", "overall-assessment"),
    ):
        assert cell_ids.index(explanation_id) < cell_ids.index(result_id)
        assert cell_ids.index(result_id) > cell_ids.index("show-summary")
