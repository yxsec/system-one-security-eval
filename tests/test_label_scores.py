"""Regression cases for reasoning-token and final-label score alignment."""

import math
from security_eval.score_alignment import extract


def entry(token, a=-2, b=-1):
    return {'token': token, 'top_logprobs': [{'token': 'A', 'logprob': a}, {'token': 'B', 'logprob': b}]}


def test_reasoning_letter_cannot_supply_final_probability():
    entries = [entry('B', -8, -1), entry('</think>'), entry('B', -1, -4), entry('<|user|>')]
    value = extract('B', {'content': entries})
    assert value['final_token_index'] == 2
    assert math.isclose(value['p_unsafe'], 1/(1+math.exp(3)))


def test_direct_answer_matches_existing_nonreasoning_readout():
    value = extract('A', {'content': [entry('A')]})
    assert value['final_token_index'] == 0
    assert math.isclose(value['p_unsafe'], 1/(1+math.exp(-1)))


def test_reasoning_and_final_text_mismatch_is_not_scored():
    assert extract('A', {'content': [entry('</think>'), entry('B')]})['p_unsafe'] is None
    assert extract('A', {'content': [entry('A'), entry(' explanation')]})['p_unsafe'] is None


def test_censored_candidate_stays_unknown():
    token = {'token': 'B', 'top_logprobs': [{'token': 'B', 'logprob': -.01}]}
    assert extract('B', {'content': [entry('</think>'), token]})['score_status'] == 'censored_candidate'


def test_sentinel_is_not_a_zero_probability():
    assert extract('A', {'content': [entry('A', -.1, -9999)]})['score_status'] == 'invalid_candidate_logprob'


def test_ambiguous_boundaries_and_invalid_labels_are_rejected():
    assert extract('B', {'content': [entry('</think>'), entry('</think>'), entry('B')]})['p_unsafe'] is None
    assert extract('B because unsafe', {'content': [entry('B')]})['p_unsafe'] is None


def test_empty_stop_normalization_keeps_the_final_position():
    from security_eval.score_normalization import extract as extract_v2
    value = extract_v2('A', {'content': [entry('B'), entry('</think>'), entry('A'), {'token':'','bytes':[]}]})
    assert value['score_status'] == 'exact' and value['final_token_index'] == 2
    assert value['empty_trailing_stop_removed']


def test_empty_token_with_nonempty_bytes_remains_ambiguous():
    from security_eval.score_normalization import extract as extract_v2
    value = extract_v2('A', {'content': [entry('A'), {'token':'','bytes':[65]}]})
    assert value['p_unsafe'] is None


def test_identical_candidate_duplicate_is_not_counted_twice():
    from security_eval.score_normalization import extract as extract_v2
    token = entry('A', -.1, -3)
    token['top_logprobs'].insert(0, dict(token['top_logprobs'][0]))
    value = extract_v2('A', {'content':[token]})
    assert value['identical_candidate_records_removed'] == 1
    assert math.isclose(value['p_unsafe'],1/(1+math.exp(2.9)))


def test_conflicting_duplicate_candidate_remains_rejected():
    from security_eval.score_normalization import extract as extract_v2
    token = entry('A', -.1, -3)
    token['top_logprobs'].insert(0, {'token':'A','logprob':-.2})
    assert extract_v2('A', {'content':[token]})['p_unsafe'] is None


def test_whitespace_after_reasoning_keeps_original_token_index():
    from security_eval.score_whitespace import extract as extract_v3
    value=extract_v3('\n\nA',{'content':[entry('B'),entry('</think>'),entry('\n\n'),entry('A'),entry('<|im_end|>')]})
    assert value['score_status']=='exact' and value['final_token_index']==3
    assert value['reasoning_end_token_index']==1
    assert value['leading_answer_whitespace_positions']==[2]


def test_whitespace_cannot_hide_explanatory_text():
    from security_eval.score_whitespace import extract as extract_v3
    value=extract_v3('A',{'content':[entry('</think>'),entry('\n'),entry('A'),entry(' because')]})
    assert value['p_unsafe'] is None
