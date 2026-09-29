"""Restore pinned public source checkouts for the evaluation harness."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPOS = {
    "rjudge": ("Lordog/R-Judge", "rjudge"),
    "wainject": ("Norrrrrrr-lyn/WAInjectBench", "wainjectbench"),
    "laya": ("NandhaKishorM/laya", "laya"),
    "decider": ("Mapika/decider", "decider"),
    "nimble": ("bespokelabsai/nimble", "nimble"),
}
locks = json.loads((ROOT / "protocol/source_lock.json").read_text())
for name, (repo, directory) in REPOS.items():
    destination = ROOT / "external" / directory
    if destination.exists():
        actual = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != locks[name]:
            raise ValueError(f"Existing checkout revision differs for {name}; refusing to overwrite it")
        continue
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            f"https://github.com/{repo}.git",
            str(destination),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(destination), "checkout", locks[name]], check=True)
