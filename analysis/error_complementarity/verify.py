"""Independently recount paired error cells using sets of released input IDs."""

from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "public_release/system-one-security-eval"
if not PUBLIC.exists():
    PUBLIC = ROOT


def main():
    report = ROOT / "results"
    data = json.loads((report / "error_complementarity.json").read_text())
    cache = defaultdict(dict)
    checks = 0
    for relative, expected in data["source_hashes"].items():
        path = PUBLIC / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if relative.startswith("data/predictions/"):
            for row in map(json.loads, gzip.decompress(path.read_bytes()).decode().splitlines()):
                if row["result"]["status"] == "ok":
                    cache[row["dataset"], row["model"]][row["id"]] = row
    for dataset, item in data["datasets"].items():
        models = data["typed"] + data["judges"] + data["specialists"][dataset]
        common = set.intersection(*(set(cache[dataset, model]) for model in models))
        assert sorted(common) == item["shared_ids"]
        for base, comparisons in item["pairs"].items():
            for other, cohorts in comparisons.items():
                a, b = cache[dataset, base], cache[dataset, other]
                for cohort, saved in cohorts.items():
                    ids = common if cohort == "shared" else set(a) & set(b)
                    assert saved["n"] == len(ids)
                    assert saved["ids_sha256"] == hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
                    for label, key in ((1, "unsafe"), (0, "benign")):
                        relevant = {i for i in ids if a[i]["label"] == label}
                        wrong_a = {i for i in relevant if a[i]["result"]["prediction"] != label}
                        wrong_b = {i for i in relevant if b[i]["result"]["prediction"] != label}
                        high = {i for i in wrong_a if max(a[i]["result"]["p_unsafe"], 1-a[i]["result"]["p_unsafe"]) >= .95}
                        expected = {"n": len(relevant), "base_errors": len(wrong_a),
                                    "counterpart_errors": len(wrong_b), "corrected": len(wrong_a - wrong_b),
                                    "introduced": len(wrong_b - wrong_a), "repeated": len(wrong_a & wrong_b),
                                    "both_correct": len(relevant - wrong_a - wrong_b),
                                    "confident_base_errors": len(high), "confident_corrected": len(high - wrong_b),
                                    "confident_repeated": len(high & wrong_b)}
                        for name, value in expected.items():
                            assert saved["classes"][key][name] == value, (dataset, base, other, cohort, key, name)
                            checks += 1
                        ratio = saved["classes"][key]["correction_fraction"]
                        assert (ratio is None and not wrong_a) or (wrong_a and math.isclose(ratio, len(wrong_a - wrong_b)/len(wrong_a)))
                        checks += 1
    result = {"status": "passed", "independent_count_checks": checks, "model_pairs": data["model_pairs"],
              "source_hashes_verified": len(data["source_hashes"]), "new_inference": False}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
