import it_eval_framework


def test_public_api_exposes_version_and_evaluate() -> None:
    assert it_eval_framework.__version__ == "0.1.3"
    assert {"evaluate", "__version__"}.issubset(it_eval_framework.__all__)
