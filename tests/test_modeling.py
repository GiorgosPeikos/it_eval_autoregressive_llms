from huggingface_hub.utils import EntryNotFoundError

from it_eval_framework.config import ModelConfig
from it_eval_framework.utils import modeling


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"


class FakeBatch(dict):
    def to(self, device):
        self["observed_device"] = device
        return self


class FakeParameter:
    device = "cpu"


class FakeModel:
    def __init__(self):
        self.moved_to = None

    def eval(self):
        return self

    def to(self, device):
        self.moved_to = device

    def parameters(self):
        yield FakeParameter()


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


def test_prepare_generation_inputs_removes_decoder_unused_segment_ids():
    observed = {}

    def tokenizer(prompt, **kwargs):
        observed["prompt"] = prompt
        observed["kwargs"] = kwargs
        return FakeBatch(input_ids=[[1, 2]], attention_mask=[[1, 1]], token_type_ids=[[0, 0]])

    encoded = modeling.prepare_generation_inputs(tokenizer, "Ciao", "cuda")

    assert observed["prompt"] == "Ciao"
    assert observed["kwargs"] == {
        "return_tensors": "pt",
        "return_token_type_ids": False,
    }
    assert "token_type_ids" not in encoded
    assert encoded["observed_device"] == "cuda"


def test_load_model_passes_device_map_without_calling_to(monkeypatch):
    observed = {}
    fake_model = FakeModel()

    def fake_from_pretrained(source, **kwargs):
        observed["source"] = source
        observed["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(modeling.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)

    result = modeling.load_model(
        ModelConfig(source="owner/large-model", device="auto", device_map="auto", max_memory={0: "20GiB"})
    )

    assert result is fake_model
    assert observed["kwargs"]["device_map"] == "auto"
    assert observed["kwargs"]["max_memory"] == {0: "20GiB"}
    assert fake_model.moved_to is None


def test_load_model_auto_selects_cuda_when_available(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(modeling.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(modeling.torch.cuda, "is_available", lambda: True)

    modeling.load_model(ModelConfig(source="owner/model", device="auto"))

    assert fake_model.moved_to == "cuda"
