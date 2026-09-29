"""Fetch selected pinned public checkpoints without changing existing model locks."""

import argparse
import getpass
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi, snapshot_download

from security_eval.common import write_json

models = {
    "laya": "convaiinnovations/laya",
    "protectai": "protectai/deberta-v3-base-prompt-injection-v2",
    "decider": "Mapika/decider-2b",
    "nimble": "bespokelabs/Bespoke-Nimble-9B",
    "piguard": "leolee99/PIGuard",
    "prompt_guard_2": "meta-llama/Llama-Prompt-Guard-2-86M",
    "wildguard": "allenai/wildguard",
    "decider_parent": "Qwen/Qwen3.5-2B-Base",
}
lock_file = ROOT / "protocol/model_lock.json"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--models", nargs="+", choices=list(models), default=["laya", "protectai"])
parser.add_argument("--prompt-token", action="store_true", help="Read a token without echoing or storing it")
args = parser.parse_args()
token_path = Path.home() / ".cache/huggingface/token"
token = (getpass.getpass("Hugging Face token: ") if args.prompt_token else os.environ.get("HF_TOKEN")
         or (token_path.read_text().strip() if token_path.exists() else None))
api = HfApi(token=token)
out = json.loads(lock_file.read_text()) if lock_file.exists() else {}
patterns_general = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py", "README.md", "LICENSE*"]
for name in args.models:
    repo = models[name]
    info = api.model_info(repo, revision=out.get(name, {}).get("revision"), files_metadata=True)
    out[name] = {**out.get(name, {}),
        "repo": repo,
        "revision": info.sha,
        "gated": info.gated,
        "pipeline_tag": info.pipeline_tag,
    }
    size = sum(f.size or 0 for f in info.siblings if f.rfilename.endswith(".safetensors"))
    print(name, info.sha, f"weights: {size / 1e9:.2f} GB", flush=True)
    patterns = (["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
                if name == "laya" else patterns_general)
    destination = ROOT / "models" / ("nimble_release" if name == "nimble" else name)
    path = snapshot_download(repo, revision=info.sha, allow_patterns=patterns,
                             local_dir=destination, max_workers=4, token=token)
    out[name]["downloaded"] = True
    out[name]["local_path"] = str(Path(path).relative_to(ROOT))
    adapter = destination / "adapter_config.json"
    out[name]["inference_ready"] = not adapter.exists()
    write_json(lock_file, out)
    if name == "nimble" and adapter.exists():
        contract = json.loads((destination / "schema_config.json").read_text())
        print("Downloading the pinned Nimble base model for the official adapter merge.", flush=True)
        base_path = snapshot_download(contract["model"], revision=contract["revision"],
                                      allow_patterns=patterns_general, local_dir=ROOT / "models/nimble_base",
                                      max_workers=4, token=token)
        out[name].update({"base_local_path": str(Path(base_path).relative_to(ROOT)),
                          "base_model": contract["model"], "base_revision": contract["revision"]})
        write_json(lock_file, out)
    print(name, info.sha, "downloaded", flush=True)
