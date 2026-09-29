"""Normalize empty stops and identical duplicate candidate records before alignment."""

import json
from .score_alignment import extract as original_extract


def extract(content, logprobs):
    entries = (logprobs or {}).get('content') or []
    empty_stop = bool(entries) and entries[-1].get('token') == '' and entries[-1].get('bytes') in (None, [])
    retained = entries[:-1] if empty_stop else entries
    normalized_entries = []
    duplicate_count = 0
    for entry in retained:
        candidates, seen = [], set()
        for candidate in entry.get('top_logprobs') or []:
            identity = json.dumps(candidate, sort_keys=True)
            if identity in seen:
                duplicate_count += 1
                continue
            seen.add(identity)
            candidates.append(candidate)
        normalized_entries.append({**entry,'top_logprobs':candidates})
    normalized = {**(logprobs or {}),'content':normalized_entries}
    score = original_extract(content, normalized)
    return {**score, 'score_aligned_to_emitted_label':'final_token_index' in score, 'empty_trailing_stop_removed': empty_stop,
            'identical_candidate_records_removed': duplicate_count}
