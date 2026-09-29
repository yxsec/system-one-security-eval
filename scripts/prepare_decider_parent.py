"""Export the pinned parent's unchanged text weights for the official Decider scorer."""

import hashlib
import json

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

from security_eval.common import ROOT, write_json


def main():
    record = json.loads((ROOT / "protocol/model_lock.json").read_text())["decider_parent"]
    source = ROOT / record["local_path"]
    destination = ROOT / "models/decider_parent_text"
    if (destination / "READY.json").exists():
        ready = json.loads((destination / "READY.json").read_text())
        if ready["revision"] != record["revision"]:
            raise ValueError("Parent export revision changed")
        print("Pinned parent text export already exists.")
        return
    if destination.exists():
        raise FileExistsError("Incomplete parent export exists; inspect before resuming")
    torch.set_num_threads(4)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        source, local_files_only=True, dtype=torch.bfloat16, device_map="cpu")
    with torch.device("meta"):
        text = Qwen3_5ForCausalLM(base.config.text_config)
    text.model, text.lm_head = base.model.language_model, base.lm_head
    text.save_pretrained(destination, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(destination)
    config = json.loads((ROOT / "models/decider/decider_config.json").read_text())
    config.update(temperature=1.0, temperature_schema_first=1.0, version="unadapted_parent_control",
                  stage="no fine-tuning; unchanged parent text weights")
    write_json(destination / "decider_config.json", config)
    hashes = {}
    for path in destination.glob("*.safetensors"):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                value.update(chunk)
        hashes[path.name] = value.hexdigest()
    write_json(destination / "READY.json", {
        "repo": record["repo"], "revision": record["revision"], "precision": "bfloat16",
        "transformation": "text submodule and LM head extracted unchanged; visual weights unused",
        "temperature": 1.0, "trained_on_evaluation_data": False, "weight_sha256": hashes,
    })
    print("Unadapted parent text export completed.")


if __name__ == "__main__":
    main()
