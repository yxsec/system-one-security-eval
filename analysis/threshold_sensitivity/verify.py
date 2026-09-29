"""Independently replay every selected policy and compare the original frozen results."""

import json
import math
from pathlib import Path

from analyze import DEFAULT_ROOT, load


def main():
    root = DEFAULT_ROOT
    protocol, grouped = load(root)
    summary = json.loads((root / "results/threshold_sensitivity.json").read_text())
    checks = 0
    for policy in summary["policies"]:
        for part in ("selection", "confirmation", "test"):
            allow = block = fn = fp = 0
            rows = grouped[(policy["dataset"], policy["model"], part)]
            for row in rows:
                p = row["result"].get("p_unsafe")
                if row["result"]["status"] != "ok" or p is None:
                    continue
                if policy["temperature"] != 1:
                    p = max(1e-7, min(1 - 1e-7, p))
                    z = math.log(p / (1 - p)) / policy["temperature"]
                    p = 1 / (1 + math.exp(-z))
                if p < policy["low"]:
                    allow += 1
                    fn += row["label"] == 1
                elif p > policy["high"]:
                    block += 1
                    fp += row["label"] == 0
            observed = policy[part]
            assert [allow, block, fn, fp] == [observed[k] for k in ("automatic_allow", "automatic_block", "unsafe_allows", "safe_blocks")]
            checks += 4
            if policy["family"] == "symmetric" and part == "confirmation":
                setting = protocol["settings"][f"{policy['dataset']}__{policy['model']}"]
                original = next(b for b in setting["original_symmetric_bands"]
                                if b["score_kind"] == policy["score"] and b["risk_target"] == policy["risk_target"])
                for k, value in original["confirmation"].items():
                    assert observed[k] == value
                    checks += 1
    public = (root / "results/heldout_current.json").exists()
    base = root if public else Path(__file__).resolve().parents[2] / "paper/fse2027"
    prefix = "results" if public else "evidence"
    original_tests = []
    for name in ("heldout_current", "probability_followup"):
        original_tests.extend(json.loads((base / prefix / f"{name}.json").read_text())["policies"])
    for p in summary["policies"]:
        if p["family"] != "symmetric":
            continue
        original = next(r for r in original_tests if all(r[k] == p[k] for k in ("dataset", "model", "score", "risk_target")))
        for k in ("low", "high", "temperature"):
            assert original[k] == p[k]
            checks += 1
        for k in ("n", "unsafe", "safe", "automatic_allow", "automatic_block", "unsafe_allows", "safe_blocks", "coverage", "allow_coverage", "miss_auto", "false_block", "allow_risk"):
            assert original[k] == p["test"][k]
            checks += 1
    print(json.dumps({"independent_count_and_original_result_checks": checks, "status": "passed"}))


if __name__ == "__main__":
    main()
