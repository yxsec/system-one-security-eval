"""Compute label and probability metrics for one newly executed experiment."""

import argparse
import json
import math
from pathlib import Path

from security_eval.common import read_jsonl, write_json
from security_eval.metrics import classification, probability_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    identities = {(r["model"], r["dataset"]) for r in rows}
    if len(identities) != 1 or len({r["id"] for r in rows}) != len(rows):
        parser.error("Provide unique inputs from one model and benchmark")
    valid = [r for r in rows if r["result"]["status"] == "ok" and r["result"].get("prediction") in (0, 1)]
    scored = [r for r in valid if isinstance(r["result"].get("p_unsafe"), (int, float))
              and math.isfinite(r["result"]["p_unsafe"]) and 0 <= r["result"]["p_unsafe"] <= 1]
    model, dataset = next(iter(identities))
    summary = {"model": model, "dataset": dataset, "requested_inputs": len(rows),
               "classification": classification([r["label"] for r in valid], [r["result"]["prediction"] for r in valid]),
               "probability": probability_metrics([r["label"] for r in scored], [r["result"]["p_unsafe"] for r in scored])}
    if args.output.exists():
        parser.error("Choose a new output path to preserve existing results")
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
