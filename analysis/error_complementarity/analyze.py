"""Measure paired error complementarity from the released prediction cache."""

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "public_release/system-one-security-eval"
if not PUBLIC.exists():
    PUBLIC = ROOT
sys.path.insert(0, str(PUBLIC / "src"))
from security_eval.paper_analysis import bootstrap_weights, interval

TYPED = ["jev", "laya", "decider", "nimble"]
JUDGES = ["gpt41", "qwen3_8b_dashscope", "glm53_flash_hosted_gateway_1", "deepseek41_flash",
          "glm53", "kimi26", "qwen38_27b"]
SPECIALISTS = {"wainject": ["protectai", "piguard", "prompt_guard_2"],
               "rjudge": [], "agentharm": ["wildguard"]}
SEED = 20260926
REPLICATES = 2000


def load_cache():
    """Validate unique IDs and labels against the released test partitions."""
    hashes = {}

    def read(relative):
        path = PUBLIC / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]

    truth = {row["id"]: row for row in read("data/partitions.jsonl.gz") if row["partition"] == "test"}
    cache = defaultdict(dict)
    for filename in ("primary_classification", "glm_completed", "additional_judges"):
        for row in read(f"data/predictions/{filename}.jsonl.gz"):
            key, record_id = (row["dataset"], row["model"]), row["id"]
            assert record_id not in cache[key]
            assert row["label"] == truth[record_id]["label"]
            assert row["dataset"] == truth[record_id]["dataset"]
            cache[key][record_id] = {**truth[record_id], **row}
    return cache, hashes


def valid_ids(rows):
    return {key for key, row in rows.items() if row["result"]["status"] == "ok"}


def paired_counts(base, counterpart, ids, with_intervals):
    """Separate corrections, new errors, and errors retained by both models."""
    records = [base[key] for key in ids]
    y = np.array([row["label"] for row in records])
    a = np.array([row["result"]["prediction"] for row in records])
    b = np.array([counterpart[key]["result"]["prediction"] for key in ids])
    probabilities = np.array([row["result"]["p_unsafe"] for row in records])
    assert np.isfinite(probabilities).all()
    confident = np.maximum(probabilities, 1 - probabilities) >= 0.95
    weights, clusters = bootstrap_weights(records, REPLICATES, SEED) if with_intervals else (None, None)
    result = {"n": len(ids), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
              "cluster_count": clusters, "classes": {}}
    for label, name in ((1, "unsafe"), (0, "benign")):
        selected = y == label
        wrong, other_wrong = selected & (a != y), selected & (b != y)
        corrected = wrong & ~other_wrong
        introduced = ~wrong & other_wrong
        repeated = wrong & other_wrong
        high = wrong & confident
        row = {"n": int(selected.sum()), "base_errors": int(wrong.sum()),
               "counterpart_errors": int(other_wrong.sum()), "corrected": int(corrected.sum()),
               "introduced": int(introduced.sum()), "repeated": int(repeated.sum()),
               "both_correct": int((selected & ~wrong & ~other_wrong).sum()),
               "confident_base_errors": int(high.sum()),
               "confident_corrected": int((high & corrected).sum()),
               "confident_repeated": int((high & repeated).sum())}
        row["correction_fraction"] = row["corrected"] / row["base_errors"] if row["base_errors"] else None
        assert row["base_errors"] == row["corrected"] + row["repeated"]
        assert row["counterpart_errors"] == row["introduced"] + row["repeated"]
        assert row["n"] == sum(row[k] for k in ("corrected", "introduced", "repeated", "both_correct"))
        if with_intervals:
            denominator = weights @ wrong
            ratio = np.divide(weights @ corrected, denominator,
                              out=np.full(REPLICATES, np.nan), where=denominator != 0)
            row["correction_fraction_interval"] = interval(ratio)
        result["classes"][name] = row
    return result


def main():
    cache, hashes = load_cache()
    output = {"scope": "Post-hoc paired error complementarity, without cascade execution or test-fitted routing",
              "reference": "https://arxiv.org/abs/2609.29769",
              "confidence": "max(p_unsafe, 1-p_unsafe) >= 0.95, matching the existing confidence diagnostic",
              "judges": JUDGES, "typed": TYPED, "specialists": SPECIALISTS,
              "score_contract": "Original primary decisions and typed probabilities, with completed GLM extension responses. No GPT/Qwen probability followups substituted for their original classification runs.",
              "bootstrap": {"replicates": REPLICATES, "seed": SEED,
                            "units": "WAInjectBench source family, R-Judge input, AgentHarm original task",
                            "interpretation": "Descriptive intervals. Sparse and zero-event cells do not certify risk."},
              "source_hashes": hashes, "new_model_inference": False, "datasets": {}}
    pairs = 0
    for dataset in ("wainject", "rjudge", "agentharm"):
        counterparts = JUDGES + SPECIALISTS[dataset]
        common = sorted(set.intersection(*(valid_ids(cache[dataset, model]) for model in TYPED + counterparts)))
        item = {"shared_ids": common, "shared_n": len(common), "pairs": {}}
        for base in TYPED:
            item["pairs"][base] = {}
            for other in counterparts:
                a, b = cache[dataset, base], cache[dataset, other]
                native_ids = sorted(valid_ids(a) & valid_ids(b))
                assert all(a[i]["label"] == b[i]["label"] for i in native_ids)
                item["pairs"][base][other] = {
                    "shared": paired_counts(a, b, common, True),
                    "pairwise_native": paired_counts(a, b, native_ids, False)}
                pairs += 1
        output["datasets"][dataset] = item
        print(f"{dataset}: {len(common)} shared inputs, {4 * len(counterparts)} model pairs", flush=True)
    output["model_pairs"] = pairs
    destination = ROOT / "results/regenerated_error_complementarity.json"
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
