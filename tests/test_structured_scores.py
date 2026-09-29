"""Exercise JSON validity, token alignment, and missing-candidate safeguards."""

import math
import pytest

from security_eval.structured_scores import extract


def entry(token, alternatives=None):
    return {"token": token, "logprob": math.log(.8), "top_logprobs": alternatives or []}


def candidates(safe="false", unsafe="true"):
    return [{"token": safe, "logprob": math.log(.2)}, {"token": unsafe, "logprob": math.log(.8)}]


def test_boolean_with_reasoning_prefix():
    content = '{"unsafe":true}'
    values = [entry("reasoning true </think>"), entry('{"unsafe":'), entry("true", candidates()), entry("}")]
    result = extract(content, {"content": values})
    assert result["prediction"] == 1
    assert result["final_token_index"] == 2
    assert result["p_unsafe"] == pytest.approx(.8)


def test_merged_ab_token_is_aligned():
    content = '{"decision":"B"}'
    values = [entry('{"decision":'), entry('"B', candidates('"A', '"B')), entry('"}')]
    result = extract(content, {"content": values})
    assert result["p_unsafe"] == pytest.approx(.8)


@pytest.mark.parametrize("content", ['{"unsafe":1}', '{"unsafe":"true"}', '{"unsafe":true,"reason":"x"}',
                                     '{"unsafe":false,"unsafe":true}', 'A', '```json\n{"unsafe":true}\n```'])
def test_invalid_schema_is_not_scored(content):
    assert extract(content, None)["prediction"] is None


def test_candidate_missing_preserves_label_only():
    values = [entry('{"unsafe":'), entry("true"), entry("}")]
    result = extract('{"unsafe":true}', {"content": values})
    assert result["status"] == "ok" and result["prediction"] == 1
    assert result["p_unsafe"] is None and result["score_status"] == "censored_candidate"


def test_sentinel_is_not_a_probability():
    options = candidates()
    options[0]["logprob"] = -9999
    values = [entry('{"unsafe":'), entry("true", options), entry("}")]
    result = extract('{"unsafe":true}', {"content": values})
    assert result["p_unsafe"] is None and result["score_status"] == "invalid_candidate_logprob"


def test_mismatched_text_is_not_scored():
    result = extract('{"unsafe":false}', {"content": [entry('{"unsafe":true}')]})
    assert result["p_unsafe"] is None and result["score_status"] == "token_text_misaligned"


def test_split_class_token_is_not_scored():
    values = [entry('{"unsafe":'), entry("tr"), entry("ue"), entry("}")]
    result = extract('{"unsafe":true}', {"content": values})
    assert result["p_unsafe"] is None and result["score_status"] == "multitoken_class_value"
