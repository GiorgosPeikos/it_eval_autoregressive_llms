from it_eval_framework.metrics.minimal_pairs import SequenceScore, choose_preferred


def test_choose_preferred_raw():
    good = SequenceScore(total_logprob=-2.0, token_count=4, char_count=20)
    bad = SequenceScore(total_logprob=-3.0, token_count=4, char_count=20)
    assert choose_preferred(good, bad, normalization="raw") == 0


def test_choose_preferred_token_normalized():
    good = SequenceScore(total_logprob=-4.0, token_count=2, char_count=10)
    bad = SequenceScore(total_logprob=-5.0, token_count=4, char_count=10)
    assert choose_preferred(good, bad, normalization="token") == 1
