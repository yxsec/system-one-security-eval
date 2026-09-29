"""Restore the published partition membership using original benchmark inputs."""

from collections import defaultdict
import gzip
import json

from security_eval.common import ROOT, git_revision, write_jsonl
from security_eval.data import agentharm_records, rjudge_records, wainject_records


def main():
    locks = json.loads((ROOT / "protocol/source_lock.json").read_text())
    for name, directory in (("rjudge", "rjudge"), ("wainject", "wainjectbench")):
        if git_revision(ROOT / "external" / directory) != locks[name]:
            raise ValueError(f"Unexpected source revision for {name}")
    records = list(rjudge_records(ROOT / "external/rjudge"))
    records += list(wainject_records(ROOT / "external/wainjectbench"))
    records += list(agentharm_records(ROOT / "data/raw", "validation"))
    records += list(agentharm_records(ROOT / "data/raw", "test_public"))
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate benchmark identifiers")
    partitions = defaultdict(list)
    with gzip.open(ROOT / "data/partitions.jsonl.gz", "rt") as stream:
        for row in map(json.loads, stream):
            original = by_id[row["id"]]
            if original["label"] != row["label"]:
                raise ValueError("Original benchmark label differs from the published partition")
            partitions[row["dataset"], row["partition"]].append({**original, **row})
    for (dataset, partition), rows in partitions.items():
        write_jsonl(ROOT / f"data/inputs/{dataset}__{partition}.jsonl", rows)
    print(json.dumps({"partitions": len(partitions), "inputs": sum(map(len, partitions.values()))}))


if __name__ == "__main__":
    main()
