"""Compare source-selected symmetric and independent thresholds on saved scores."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE if (HERE / "protocol").exists() else HERE.parents[1]


def arrays(rows, temperature):
    y = np.asarray([r["label"] for r in rows], dtype=int)
    p = np.asarray([r["result"].get("p_unsafe") if r["result"]["status"] == "ok" else None for r in rows], dtype=float)
    assert np.all(np.isnan(p) | ((p >= 0) & (p <= 1)))
    if temperature != 1:
        p = expit(logit(np.clip(p, 1e-7, 1 - 1e-7)) / temperature)
    return y, p


def metrics(y, p, low, high):
    assert 0 <= low <= 0.5 <= high <= 1
    allow, block = p < low, p > high
    n, unsafe, safe = len(y), int(sum(y)), int(sum(y == 0))
    na, nb = int(sum(allow)), int(sum(block))
    fn, fp = int(sum(allow & (y == 1))), int(sum(block & (y == 0)))
    return dict(n=n, unsafe=unsafe, safe=safe, scored=int(np.isfinite(p).sum()),
                automatic_allow=na, automatic_block=nb, unsafe_allows=fn, safe_blocks=fp,
                coverage=(na + nb) / n, allow_coverage=na / n,
                miss_auto=fn / unsafe if unsafe else None, false_block=fp / safe if safe else None,
                allow_risk=fn / na if na else None)


def passes(value, miss_budget, block_budget):
    return (value["miss_auto"] is not None and value["false_block"] is not None
            and value["miss_auto"] <= miss_budget and value["false_block"] <= block_budget)


def select(rows, temperature, miss_budget, block_budget, family):
    """Select thresholds using only this source partition, with deterministic ties."""
    y, p = arrays(rows, temperature)
    highs = np.linspace(0.5, 1, 101)
    lows = 1 - highs
    if family == "symmetric":
        candidates = [{"low": float(low), "high": float(high), **metrics(y, p, low, high)}
                      for low, high in zip(lows, highs)]
    elif family == "independent":
        # The two tails never overlap, so feasible maximum counts can be chosen separately.
        allow_options = [(float(low), metrics(y, p, low, 1.0)) for low in lows]
        block_options = [(float(high), metrics(y, p, 0.0, high)) for high in highs]
        feasible_allow = [(low, value) for low, value in allow_options if passes(value, miss_budget, block_budget)]
        feasible_block = [(high, value) for high, value in block_options if passes(value, miss_budget, block_budget)]
        low, _ = min(feasible_allow, key=lambda item: (-item[1]["automatic_allow"], item[0]))
        high, _ = min(feasible_block, key=lambda item: (-item[1]["automatic_block"], -item[0]))
        candidates = [{"low": low, "high": high, **metrics(y, p, low, high)}]
    else:
        raise ValueError(f"Unknown threshold family: {family}")
    feasible = [c for c in candidates if passes(c, miss_budget, block_budget)]
    return min(feasible, key=lambda c: (-c["coverage"], -c["allow_coverage"], -c["high"], c["low"]))


def load(root):
    protocol_path = root / "protocol/threshold_sensitivity.json"
    protocol = json.loads(protocol_path.read_text())
    path = root / protocol["observation_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol["observation_sha256"]
    rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["partition"])].append(row)
    for dataset in protocol["datasets"]:
        for model in protocol["models"]:
            sets = [{r["id"] for r in grouped[(dataset, model, part)]} for part in ("selection", "confirmation", "test")]
            assert all(sets) and not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    return protocol, grouped


def source_policies(protocol, grouped):
    selected = []
    for dataset in protocol["datasets"]:
        for model in protocol["models"]:
            setting = protocol["settings"][f"{dataset}__{model}"]
            for score in protocol["scores"]:
                temperature = 1 if score == "shipped" else setting["temperature"]
                for budget in protocol["miss_budgets"]:
                    for family in protocol["families"]:
                        chosen = select(grouped[(dataset, model, "selection")], temperature, budget,
                                        protocol["false_block_budget"], family)
                        if family == "symmetric":
                            original = next(b for b in setting["original_symmetric_bands"]
                                            if b["score_kind"] == score and b["risk_target"] == budget)
                            for key, value in original["selected"].items():
                                assert chosen[key] == value, (dataset, model, score, budget, key)
                        selected.append(dict(dataset=dataset, model=model, score=score, temperature=temperature,
                                             risk_target=budget, false_block_budget=protocol["false_block_budget"],
                                             family=family, low=chosen.pop("low"), high=chosen.pop("high"), selection=chosen))
    assert len(selected) == 144
    return {"protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
            "selection_uses_test_or_confirmation": False, "policies": selected}


def paired_gain(rows, temperature, symmetric, independent, seed, replicates):
    _, p = arrays(rows, temperature)
    auto_s = (p < symmetric["low"]) | (p > symmetric["high"])
    auto_i = (p < independent["low"]) | (p > independent["high"])
    difference = auto_i.astype(float) - auto_s.astype(float)
    groups = [r["family"] if r["dataset"] == "wainject" else r["group"] for r in rows]
    unique = sorted(set(groups))
    sums = np.array([difference[np.array(groups) == g].sum() for g in unique])
    sizes = np.array([groups.count(g) for g in unique])
    draws = np.random.default_rng(seed).integers(0, len(unique), size=(replicates, len(unique)))
    values = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return {"estimate": float(difference.mean()), "ci95": np.quantile(values, [.025, .975]).tolist(), "clusters": len(unique)}


def evaluate(protocol, grouped, policies):
    results = []
    for selected in policies["policies"]:
        cell = dict(selected)
        for part in ("confirmation", "test"):
            y, p = arrays(grouped[(cell["dataset"], cell["model"], part)], cell["temperature"])
            value = metrics(y, p, cell["low"], cell["high"])
            cell[part] = {**value, "budgets_met": passes(value, cell["risk_target"], cell["false_block_budget"])}
        results.append(cell)
    contrasts = []
    for independent in [r for r in results if r["family"] == "independent"]:
        same = ("dataset", "model", "score", "risk_target")
        symmetric = next(r for r in results if r["family"] == "symmetric" and all(r[k] == independent[k] for k in same))
        assert independent["selection"]["coverage"] >= symmetric["selection"]["coverage"]
        rows = grouped[(independent["dataset"], independent["model"], "test")]
        interval = paired_gain(rows, independent["temperature"], symmetric, independent,
                               protocol["bootstrap"]["seed"], protocol["bootstrap"]["replicates"])
        contrasts.append({**{k: independent[k] for k in same}, "coverage_gain": interval,
                          "selection_gain": independent["selection"]["coverage"] - symmetric["selection"]["coverage"],
                          "symmetric_confirmation_met": symmetric["confirmation"]["budgets_met"],
                          "independent_confirmation_met": independent["confirmation"]["budgets_met"],
                          "symmetric_test_met": symmetric["test"]["budgets_met"],
                          "independent_test_met": independent["test"]["budgets_met"]})
    return {"scope": protocol["purpose"], "new_inference": False, "policy_count": len(results),
            "observation_sha256": protocol["observation_sha256"], "policies": results, "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    protocol, grouped = load(args.root)
    out = args.root / "results"
    policy_path = out / "threshold_policies.json"
    result_path = out / "threshold_sensitivity.json"
    selected = source_policies(protocol, grouped)
    if args.select:
        out.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(json.dumps(selected, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"source_policies": len(selected["policies"]), "original_symmetric_policies_reproduced": 72}))
        return
    assert selected == json.loads(policy_path.read_text()), "Source-policy file differs from the selection-only calculation"
    summary = evaluate(protocol, grouped, selected)
    if args.check:
        assert summary == json.loads(result_path.read_text()), "Saved threshold results differ"
    else:
        result_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "policies": len(summary["policies"]), "paired_contrasts": len(summary["contrasts"]), "new_inference": False}))


if __name__ == "__main__":
    main()
