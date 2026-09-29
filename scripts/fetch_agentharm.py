"""Fetch the pinned original AgentHarm public files for the availability audit."""

import httpx

from security_eval.common import ROOT, write_json

REVISION = "e23b3fe60a0da9037314b88e5ee3a0c054970dad"
REPO = "ai-safety-institute/AgentHarm"
destination = ROOT / "data/raw"
destination.mkdir(parents=True, exist_ok=True)
with httpx.Client(timeout=60, follow_redirects=True) as client:
    response = client.get(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}")
    response.raise_for_status()
    write_json(destination / "agentharm_metadata.json", response.json())
    for label in ("benign", "harmful"):
        for split in ("test_public", "validation"):
            name = f"{label}_behaviors_{split}.json"
            response = client.get(
                f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/benchmark/{name}"
            )
            response.raise_for_status()
            write_json(destination / f"agentharm_{name}", response.json())
print("Pinned AgentHarm public artifacts saved; no examples or labels were generated.")
