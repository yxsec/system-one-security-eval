"""Verify final probability results from selected responses, without inference."""

from collections import defaultdict
import argparse
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from security_eval.metrics import classification, probability_metrics


def quality(records):
    return {
        "classification": classification([r["label"] for r in records], [r["result"]["prediction"] for r in records]),
        "probability": probability_metrics([r["label"] for r in records], [r["result"]["p_unsafe"] for r in records]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the distributed final summary.")
    args = parser.parse_args()
    with gzip.open(ROOT / "data/predictions/completed_probabilities.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == len({(r["model"], r["id"]) for r in records}) == 6600
    cells = defaultdict(list)
    for row in records:
        result = row["result"]
        assert result["status"] == "ok" and result["prediction"] in (0, 1)
        assert math.isfinite(result["p_unsafe"]) and 0 <= result["p_unsafe"] <= 1
        cells[row["dataset"], row["model"]].append(row)
    saved = json.loads((ROOT / "results/completed_probability.json").read_text())
    count = 0
    for dataset, models in saved["datasets"].items():
        for model, expected in models.items():
            native = cells[dataset, model]
            shared = [row for row in native if row["id"] in set(expected["shared_ids"])]
            assert len(shared) == expected["shared_n"]
            assert quality(native) == expected["native"]
            assert quality(shared) == expected["shared"]
            without_json = [r for r in shared if r.get("recovery_kind") != "json_ab_fallback"]
            assert quality(without_json) == expected["without_json_shared"]
            count += len(shared)
    print(json.dumps({"status": "passed", "completed_probability_cells": len(cells),
                      "selected_responses": len(records), "shared_scores": count,
                      "new_inference": False}))


if __name__ == "__main__":
    main()
