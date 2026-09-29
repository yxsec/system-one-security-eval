"""Run Nimble's official local adapter merge and record the resulting model identity."""

import json
import os
import subprocess
import sys

from security_eval.common import ROOT, write_json

lock_path = ROOT / "protocol/model_lock.json"
lock = json.loads(lock_path.read_text())
record = lock["nimble"]
release = ROOT / "models/nimble_release"
output = ROOT / "models/nimble"
env = dict(os.environ, PYTHONPATH=str(ROOT / "external/nimble"), HF_HUB_OFFLINE="1")
subprocess.run([sys.executable, "-m", "nimble.scoring.merge_local_adapter", "--adapter", str(release),
                "--output", str(output), "--base", str(ROOT / record["base_local_path"])],
               check=True, env=env, cwd=ROOT)
ready = json.loads((output / "READY.json").read_text())
if ready["base_revision"] != record["base_revision"]:
    raise ValueError("Merged base revision differs from the model lock")
record.update(local_path=str(output.relative_to(ROOT)), release_path=str(release.relative_to(ROOT)),
              inference_ready=True, adapter_sha256=ready["adapter_sha256"])
write_json(lock_path, lock)
print("Verified merged Nimble checkpoint is ready for inference.")
