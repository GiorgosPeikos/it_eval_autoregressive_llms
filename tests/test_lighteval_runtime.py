from it_eval_framework.utils import lighteval_runtime


def test_preflight_accepts_pinned_runtime(monkeypatch):
    versions = {
        "lighteval": "0.13.0",
        "datasets": "3.6.0",
        "transformers": "4.57.1",
        "accelerate": "1.10.0",
    }
    monkeypatch.setattr(lighteval_runtime, "installed_version", versions.get)

    report = lighteval_runtime.lighteval_environment_report()

    assert report["errors"] == []
    assert report["warnings"] == []


def test_preflight_rejects_incompatible_lighteval_and_datasets(monkeypatch):
    versions = {
        "lighteval": "0.14.0",
        "datasets": "4.0.0",
        "transformers": "4.57.1",
        "accelerate": "1.10.0",
    }
    monkeypatch.setattr(lighteval_runtime, "installed_version", versions.get)

    report = lighteval_runtime.lighteval_environment_report()

    assert any("lighteval==0.13.0" in error for error in report["errors"])
    assert any("datasets==3.6.0" in error for error in report["errors"])
