import math

from it_eval_framework.metrics.perplexity import finalize_perplexity, sliding_windows


def test_sliding_windows():
    assert list(sliding_windows(10, 4, 3)) == [(0, 4), (3, 7), (6, 10)]


def test_finalize_perplexity():
    payload = finalize_perplexity(6.0, 3)
    assert payload["mean_loss"] == 2.0
    assert math.isclose(payload["token_perplexity"], math.exp(2.0))
