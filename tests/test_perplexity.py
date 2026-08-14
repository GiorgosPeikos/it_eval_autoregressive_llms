import math

import torch

from it_eval_framework.metrics.perplexity import finalize_perplexity, sliding_windows
from it_eval_framework.metrics.perplexity import compute_window_nll


def test_sliding_windows():
    assert list(sliding_windows(10, 4, 3)) == [(0, 4), (3, 7), (6, 10)]


def test_finalize_perplexity():
    payload = finalize_perplexity(6.0, 3)
    assert payload["mean_loss"] == 2.0
    assert math.isclose(payload["token_perplexity"], math.exp(2.0))


class _RecordingModel:
    def __init__(self):
        self.labels = []

    def __call__(self, input_ids, labels):
        self.labels.append(labels.clone())
        return type("Output", (), {"loss": torch.tensor(2.0)})()


def test_overlapping_window_scores_only_new_target_tokens():
    model = _RecordingModel()
    ids = torch.arange(8).unsqueeze(0)

    first = compute_window_nll(model, ids, 0, 6, target_start=0)
    second = compute_window_nll(model, ids, 3, 8, target_start=6)

    assert first.token_count == 5
    assert second.token_count == 2
    assert second.negative_log_likelihood == 4.0
    assert model.labels[1].tolist() == [[-100, -100, -100, 6, 7]]
