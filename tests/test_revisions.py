from types import SimpleNamespace

from it_eval_framework.config import EvaluationConfig
from it_eval_framework.presets import build_evaluation_config
from it_eval_framework.utils.revisions import resolve_model_revisions, resolve_repo_revision


class FakeApi:
    def __init__(self, revisions):
        self.revisions = revisions
        self.calls = []

    def model_info(self, repo_id, revision):
        self.calls.append((repo_id, revision))
        return SimpleNamespace(sha=self.revisions[(repo_id, revision)])


def config(model):
    return EvaluationConfig.model_validate(
        {
            "model": model,
            "lighteval": {"enabled": False, "suite": None},
            "blimp_it": {"enabled": False},
            "perplexity": None,
            "generation": {"enabled": False},
        }
    )


def test_resolves_same_repo_once_and_reuses_model_sha():
    api = FakeApi({("owner/model", "main"): "a" * 40})

    resolved = resolve_model_revisions(
        config({"source": "owner/model", "tokenizer_source": "owner/model"}),
        api,
    )

    assert resolved.model.revision == "a" * 40
    assert resolved.model.tokenizer_revision == "a" * 40
    assert api.calls == [("owner/model", "main")]


def test_resolves_different_repositories_independently():
    api = FakeApi(
        {
            ("owner/model", "main"): "a" * 40,
            ("owner/tokenizer", "main"): "b" * 40,
        }
    )

    resolved = resolve_model_revisions(
        config({"source": "owner/model", "tokenizer_source": "owner/tokenizer"}),
        api,
    )

    assert resolved.model.revision == "a" * 40
    assert resolved.model.tokenizer_revision == "b" * 40


def test_full_commit_sha_is_already_resolved_without_network():
    assert resolve_repo_revision("owner/model", "A" * 40, FakeApi({})) == "a" * 40


def test_preset_does_not_copy_model_revision_to_different_tokenizer_repo():
    result = build_evaluation_config(
        "owner/model",
        tokenizer="owner/tokenizer",
        revision="a" * 40,
    )

    assert result.model.tokenizer_revision is None
