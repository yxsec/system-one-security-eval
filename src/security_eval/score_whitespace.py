"""Permit whitespace surrounding an otherwise exact, uniquely located final label."""

from .score_normalization import extract as previous_extract


def extract(content, logprobs):
    entries=(logprobs or {}).get('content') or []
    closes=[i for i,entry in enumerate(entries) if entry.get('token')=='</think>']
    start=closes[-1]+1 if closes else 0
    removed=set()
    cursor=start
    while cursor<len(entries) and isinstance(entries[cursor].get('token'),str) and entries[cursor]['token'].isspace():
        removed.add(cursor)
        cursor+=1
    mapping=[i for i in range(len(entries)) if i not in removed]
    normalized={**(logprobs or {}),'content':[entries[i] for i in mapping]}
    result=previous_extract(content,normalized)
    for key in ('final_token_index','reasoning_end_token_index'):
        if result.get(key) is not None:
            result[key]=mapping[result[key]]
    return {**result,'leading_answer_whitespace_positions':sorted(removed)}
