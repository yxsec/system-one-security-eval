"""Verify the complete Laya configuration and its development-only calibration."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]


def read_rows(name):
    with gzip.open(ROOT / f"data/predictions/{name}.jsonl.gz", "rt") as stream:
        return [json.loads(line) for line in stream]


def main():
    rows = read_rows("laya_all_partitions")
    assert len(rows) == len({r["id"] for r in rows}) == 4307
    assert all(r["model"] == "laya" and r["result"]["status"] == "ok" for r in rows)
    by_id = {r["id"]: r for r in rows}
    config = json.loads((ROOT / "protocol/models.json").read_text())
    assert config["laya"]["max_len_override"] == 2048
    matched = 0
    for filename in ("primary_classification", "threshold_sensitivity"):
        for row in read_rows(filename):
            if row["model"] != "laya":
                continue
            observed = by_id[row["id"]]
            for key in ("label", "input_sha256", "dataset"):
                assert observed[key] == row[key]
            for key in ("status", "prediction", "p_unsafe", "score_kind"):
                assert observed["result"][key] == row["result"][key]
            matched += 1
    policies = json.loads((ROOT / "protocol/offline_policies.json").read_text())["policies"]
    partitions = defaultdict(list)
    for row in rows:
        partitions[row["dataset"], row["partition"]].append(row)
    for dataset in ("wainject", "rjudge", "agentharm"):
        development = partitions[dataset, "development"]
        y = np.array([r["label"] for r in development])
        logits = logit(np.clip([r["result"]["p_unsafe"] for r in development], 1e-7, 1-1e-7))

        def loss(log_temperature):
            z = logits / np.exp(log_temperature)
            return float(np.mean(np.logaddexp(0, z) - y*z))

        result = minimize_scalar(loss, bounds=(np.log(.05), np.log(20)), method="bounded")
        fitted = policies[f"{dataset}__laya"]["calibration"]
        assert result.success and fitted["n"] == len(development)
        assert np.isclose(np.exp(result.x), fitted["temperature"], rtol=1e-10)
        assert np.isclose(result.fun, fitted["development_nll_after"], rtol=1e-10)
    print(json.dumps({"status": "passed", "observations": len(rows), "linked_analysis_rows": matched,
                      "test_inputs": sum(r["partition"] == "test" for r in rows), "development_temperature_fits": 3}))


if __name__ == "__main__":
    main()
