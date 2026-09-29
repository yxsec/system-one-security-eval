"""Audit calibration binning and prevalence baselines using fixed test scores."""

from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
PUBLIC = ROOT / "public_release/system-one-security-eval"
if not PUBLIC.exists():
    PUBLIC = ROOT
MODELS = ("jev", "laya", "decider", "nimble")
NAMES = {"jev": "Jev", "laya": "Laya", "decider": "Decider", "nimble": "Nimble"}
DATASETS = ("wainject", "rjudge", "agentharm")
TASKS = {"wainject": "Prompt injection", "rjudge": "Interaction risk", "agentharm": "Harmful request"}


def bins(y, p, boundaries):
    """Keep equal scores together, including scores on a quantile boundary."""
    groups = defaultdict(list)
    for label, score in zip(y, p, strict=True):
        groups[int(np.searchsorted(boundaries, score, side="right"))].append((label, score))
    cells = [{"n": len(items), "mean_probability": sum(p for _, p in items) / len(items),
              "unsafe_rate": sum(y for y, _ in items) / len(items)} for items in groups.values()]
    ece = sum(cell["n"] * abs(cell["mean_probability"] - cell["unsafe_rate"]) for cell in cells) / len(y)
    return {"ece": ece, "occupied_bins": len(cells), "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the saved sensitivity without writing files.")
    args = parser.parse_args()
    cache = PUBLIC / "data/predictions/primary_classification.jsonl.gz"
    evidence_path = PUBLIC / "results/heldout_current.json"
    evidence = json.loads(evidence_path.read_text())
    by_cell = defaultdict(list)
    with gzip.open(cache, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["model"] in MODELS:
                by_cell[row["dataset"], row["model"]].append(row)
    cells = []
    checks = 0
    for dataset in DATASETS:
        common = set(evidence["datasets"][dataset]["common_ids"])
        for model in MODELS:
            rows = [row for row in by_cell[dataset, model] if row["id"] in common]
            assert len(rows) == len(common) and all(row["result"]["status"] == "ok" for row in rows)
            y = np.array([row["label"] for row in rows], dtype=float)
            p = np.array([row["result"]["p_unsafe"] for row in rows], dtype=float)
            assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
            methods = {f"equal_width_{count}": bins(y, p, np.arange(1, count) / count) for count in (5, 10, 20)}
            methods["quantile_10_ties_preserved"] = bins(y, p, np.unique(np.quantile(p, np.arange(1, 10) / 10)))
            brier = float(np.mean((p - y) ** 2))
            prevalence = float(np.mean(y))
            constant_brier = prevalence * (1 - prevalence)
            expected = evidence["datasets"][dataset]["models"][model]["probability_common"]
            assert math.isclose(brier, expected["brier"], abs_tol=1e-12)
            assert math.isclose(methods["equal_width_10"]["ece"], expected["ece_10_equal_width"], abs_tol=1e-12)
            assert math.isclose(float(np.mean((prevalence - y) ** 2)), constant_brier, abs_tol=1e-12)
            for method in methods.values():
                assert sum(cell["n"] for cell in method["cells"]) == len(rows)
                assert method["ece"] + 1e-12 >= abs(float(np.mean(p - y)))
            checks += 11
            cells.append({"dataset": dataset, "model": model, "n": len(rows),
                          "unsafe_n": int(np.sum(y)), "unsafe_rate": prevalence,
                          "mean_probability": float(np.mean(p)), "signed_bias": float(np.mean(p - y)),
                          "brier": brier, "constant_prevalence_brier": constant_brier,
                          "brier_skill_vs_empirical_prevalence": 1 - brier / constant_brier,
                          "binning": methods})
    summary = {"scope": "Post-hoc descriptive sensitivity on the original shared test populations, four typed components",
               "new_inference": False, "model_or_policy_fitting": False,
               "reference": "The constant empirical test prevalence is a hindsight diagnostic reference, not a fitted deployment predictor.",
               "quantile_rule": "Ten nominal quantile bins, deduplicated boundaries, with exact score ties retained in one bin. Occupied bins may be fewer than ten.",
               "source_sha256": {str(path.relative_to(PUBLIC)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (cache, evidence_path)},
               "independent_checks": checks, "cells": cells}
    if args.check:
        saved = json.loads((OUT / "calibration_sensitivity.json").read_text())
        assert saved == summary
        print(json.dumps({"status": "passed", "cells": len(cells), "independent_checks": checks,
                          "saved_results_reproduced": True, "new_inference": False}))
        return
    (OUT / "calibration_sensitivity.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    flat = [{**{k: v for k, v in cell.items() if k != "binning"},
             **{name + "_ece": value["ece"] for name, value in cell["binning"].items()},
             "quantile_occupied_bins": cell["binning"]["quantile_10_ties_preserved"]["occupied_bins"]} for cell in cells]
    with (OUT / "calibration_sensitivity.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"cells": len(cells), "checks": checks,
                      "range": [{"dataset": c["dataset"], "model": c["model"],
                                 "ece": [round(v["ece"], 4) for v in c["binning"].values()],
                                 "bias": round(c["signed_bias"], 4),
                                 "skill": round(c["brier_skill_vs_empirical_prevalence"], 4)} for c in cells]}, indent=2))


if __name__ == "__main__":
    main()
