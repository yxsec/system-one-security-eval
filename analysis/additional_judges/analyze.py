"""Analyze the collected judge extension from text-free, linked observations."""

from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
PUBLIC = ROOT / "public_release/system-one-security-eval"
if not PUBLIC.exists():
    PUBLIC = ROOT
sys.path.insert(0, str(ROOT / "src"))
from security_eval.metrics import classification, probability_metrics
from security_eval.paper_analysis import cluster_intervals, paired_comparison, holm_adjust

NAMES = {"deepseek41_flash": "DeepSeek-V4.1-Flash", "glm53": "GLM-5.3", "kimi26": "Kimi-K2.6",
         "qwen38_27b": "Qwen3.8-27B", "glm53_flash_hosted_gateway_1": "GLM-5.3-Flash",
         "jev": "Jev", "nimble": "Nimble", "gpt41": "GPT-4.1"}
ADDED = ("deepseek41_flash", "glm53", "kimi26", "qwen38_27b")


def main():
    source_hashes = {}

    def load(relative):
        path = PUBLIC / relative
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == ".gz":
            return [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
        return json.loads(path.read_text())

    original = load("results/heldout_current.json")
    cache = defaultdict(list)
    for name in ("primary_classification", "glm_completed", "additional_judges"):
        for row in load(f"data/predictions/{name}.jsonl.gz"):
            cache[row["dataset"], row["model"]].append(row)
    checker_spec = importlib.util.spec_from_file_location("independent_metrics", PUBLIC / "reproduce_results.py")
    checker = importlib.util.module_from_spec(checker_spec)
    checker_spec.loader.exec_module(checker)
    checks = 0

    def quality(rows):
        nonlocal checks
        valid = [row for row in rows if row["result"]["status"] == "ok"]
        scored = [row for row in valid if row["result"].get("p_unsafe") is not None]
        c = classification([row["label"] for row in valid], [row["result"]["prediction"] for row in valid])
        p = probability_metrics([row["label"] for row in scored], [row["result"]["p_unsafe"] for row in scored])
        for actual, expected in ((checker.classify(valid), c), (checker.probability(valid), p)):
            for key, value in actual.items():
                assert (value is None and expected[key] is None) or math.isclose(value, expected[key], rel_tol=1e-10, abs_tol=1e-12), key
                checks += 1
        return {"classification": c, "probability": p}

    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Post-protocol full-test judge extension, with original primary rankings and frozen policies preserved",
              "selection": "First complete output, independent of prediction correctness. Qwen includes one recorded JSON A/B fallback.",
              "new_inference": False, "test_fitted_parameters": False, "datasets": {},
              "source_hashes": source_hashes}
    for dataset in ("wainject", "rjudge", "agentharm"):
        common = set(original["datasets"][dataset]["common_ids"])
        assert common <= set(original["datasets"][dataset]["common_ids"])
        models = {}
        comparisons = []
        for model, name in NAMES.items():
            rows = cache[dataset, model]
            matched = [row for row in rows if row["id"] in common]
            assert len(matched) == len(common)
            assert all(row["result"]["status"] == "ok" for row in matched)
            models[model] = {"name": name, "native": quality(rows), "shared": quality(matched),
                             "shared_intervals": cluster_intervals(matched),
                             "format_fallbacks": sum(row.get("recovery_kind") == "json_ab_fallback" for row in rows)}
            if model in ADDED:
                comparison = paired_comparison(rows, cache[dataset, "jev"])
                comparison["inference_scope"] = "Exploratory post-protocol test comparison, added judge minus Jev"
                comparisons.append({"model": model, **comparison})
        for endpoint in ("fnr", "fpr"):
            adjusted = holm_adjust([row["metrics"][endpoint]["cluster_randomization_p"] for row in comparisons])
            for row, p in zip(comparisons, adjusted, strict=True):
                row["metrics"][endpoint]["holm_p_added_four_family"] = p
        order = sorted(models, key=lambda key: (-models[key]["shared"]["classification"]["macro_f1"], key))
        result["datasets"][dataset] = {"shared_ids": sorted(common), "shared_n": len(common),
                                      "models": models, "shared_order": order,
                                      "comparisons_added_minus_jev": comparisons}
        print(f"Analyzed {dataset}: {len(common)} shared inputs, {len(models)} configurations", flush=True)
    result["independent_metric_checks"] = checks
    destination = OUT / "regenerated_added_models.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Passed {checks} independent metric checks", flush=True)


if __name__ == "__main__":
    main()
