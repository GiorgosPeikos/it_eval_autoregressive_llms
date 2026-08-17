import pytest
from pathlib import Path

from it_eval_framework.presets import PRESET_NAMES, build_evaluation_config


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_all_presets_build(preset):
    config = build_evaluation_config("owner/model", preset=preset, device="cpu")
    assert config.model.source == "owner/model"


def test_full_is_unbounded_and_quick_is_bounded():
    full = build_evaluation_config("owner/model", preset="full", device="cpu")
    quick = build_evaluation_config("owner/model", preset="quick", device="cpu")

    assert full.lighteval.enabled is True
    assert full.perplexity.max_documents is None
    assert full.perplexity.dataset_streaming is True
    assert quick.lighteval.enabled is False
    assert quick.perplexity.max_documents == 3
    assert Path(quick.generation.prompts_path).exists()
