from huggingface_hub.utils import EntryNotFoundError

from it_eval_framework.config import ModelConfig
from it_eval_framework.utils import modeling


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"


def test_load_tokenizer_passes_resolved_hub_tokenizer_file(monkeypatch):
    observed = {}
    cached_file = "/cache/snapshot/tokenizer.json"
    monkeypatch.setattr(modeling, "hf_hub_download", lambda **kwargs: cached_file)

    def fake_from_pretrained(tokenizer_id, **kwargs):
        observed["tokenizer_id"] = tokenizer_id
        observed["kwargs"] = kwargs
        return FakeTokenizer()

    monkeypatch.setattr(modeling.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    modeling.load_tokenizer(
        ModelConfig(
            source="owner/model",
            tokenizer_source="owner/tokenizer",
            tokenizer_revision="a" * 40,
        )
    )

    assert observed["tokenizer_id"] == "owner/tokenizer"
    assert observed["kwargs"]["tokenizer_file"] == cached_file
    assert observed["kwargs"]["revision"] == "a" * 40


def test_resolve_tokenizer_file_uses_absolute_local_path(tmp_path):
    tokenizer_directory = tmp_path / "tokenizer"
    tokenizer_directory.mkdir()
    tokenizer_file = tokenizer_directory / "tokenizer.json"
    tokenizer_file.write_text("{}", encoding="utf-8")

    resolved = modeling.resolve_tokenizer_file(
        ModelConfig(source="owner/model", tokenizer_source=str(tokenizer_directory))
    )

    assert resolved == str(tokenizer_file.resolve())


def test_resolve_tokenizer_file_allows_repositories_without_fast_file(monkeypatch):
    def missing_file(**kwargs):
        raise EntryNotFoundError("missing tokenizer.json")

    monkeypatch.setattr(modeling, "hf_hub_download", missing_file)

    resolved = modeling.resolve_tokenizer_file(ModelConfig(source="owner/model"))

    assert resolved is None
