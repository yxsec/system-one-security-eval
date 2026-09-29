"""Read aligned binary class-token probabilities from a strict JSON decision."""

import json
import math
import re


def extract(content, logprobs):
    result = {"prediction": None, "p_unsafe": None, "score_kind": None,
              "score_status": "invalid_structured_decision", "score_aligned_to_emitted_label": False}
    if not isinstance(content, str):
        return result
    try:
        decoded = json.loads(content)
    except (ValueError, TypeError):
        return result
    if not isinstance(decoded, dict):
        return result
    if set(decoded) == {"unsafe"} and isinstance(decoded["unsafe"], bool):
        prediction = int(decoded["unsafe"])
        match = re.fullmatch(r'\s*\{\s*"unsafe"\s*:\s*(true|false)\s*\}\s*', content)
        safe, unsafe = "false", "true"
    elif set(decoded) == {"decision"} and decoded["decision"] in ("A", "B"):
        prediction = int(decoded["decision"] == "B")
        match = re.fullmatch(r'\s*\{\s*"decision"\s*:\s*"(A|B)"\s*\}\s*', content)
        safe, unsafe = "A", "B"
    else:
        return result
    if match is None:
        return result
    result.update(status="ok", prediction=prediction, score_status="missing_logprobs")
    entries = (logprobs or {}).get("content") or []
    if not entries or any(not isinstance(entry.get("token"), str) for entry in entries):
        return result
    joined = "".join(entry["token"] for entry in entries)
    if not joined.endswith(content):
        return {**result, "score_status": "token_text_misaligned"}
    offset = len(joined) - len(content)
    start, end = offset + match.start(1), offset + match.end(1)
    position = 0
    selected = None
    for index, entry in enumerate(entries):
        next_position = position + len(entry["token"])
        if position <= start and end <= next_position:
            selected = index, entry, start - position, end - position
            break
        position = next_position
    if selected is None:
        return {**result, "score_status": "multitoken_class_value"}
    index, entry, local_start, local_end = selected
    prefix, suffix = entry["token"][:local_start], entry["token"][local_end:]
    targets = {0: prefix + safe + suffix, 1: prefix + unsafe + suffix}
    values = {}
    candidates = list(entry.get("top_logprobs") or []) + [entry]
    for label, token in targets.items():
        found = [candidate.get("logprob") for candidate in candidates if candidate.get("token") == token]
        if found and any(value != found[0] for value in found):
            return {**result, "score_status": "conflicting_candidate_scores"}
        if found:
            values[label] = found[0]
    result.update(final_token_index=index, score_aligned_to_emitted_label=True,
                  candidate_tokens=targets, candidate_logprobs=values)
    if set(values) != {0, 1}:
        return {**result, "score_status": "censored_candidate"}
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or not -9990 < value <= 0 for value in values.values()):
        return {**result, "score_status": "invalid_candidate_logprob"}
    difference = values[1] - values[0]
    score = 1 / (1 + math.exp(-difference)) if difference >= 0 else math.exp(difference) / (1 + math.exp(difference))
    return {**result, "p_unsafe": score, "score_status": "exact",
            "score_kind": "structured_candidate_token_conditional"}
