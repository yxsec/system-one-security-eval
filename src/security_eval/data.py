"""Restore original benchmark inputs and labels without resampling."""

import json
from .common import digest

def rjudge_records(root):
    for path in sorted((root / "data").rglob("*.json")):
        for row in json.loads(path.read_text()):
            # Allowlist input fields. Never expose goal/label/risk_description/attack_type.
            state = json.dumps(
                {"profile": row.get("profile", ""), "interaction": row["contents"]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            label = row["label"]
            if label not in (0, 1):
                raise ValueError("Nonbinary R-Judge label")
            yield {
                "id": f"rjudge:{path.stem}:{row['id']}",
                "dataset": "rjudge",
                "task": "interaction_risk",
                "input": state,
                "label": label,
                "stratum": f"{path.stem}:{label}",
                "group": digest(state),
                "source": str(path.relative_to(root)),
                "source_id": str(row["id"]),
                "label_origin": "original label field",
                "family": path.stem,
            }


def wainject_records(root):
    for path in sorted((root / "data/text").rglob("*.jsonl")):
        label = {"benign": 0, "malicious": 1}[path.parent.name]
        for line, raw in enumerate(path.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            state = row["text"]
            if not isinstance(state, str) or not state.strip():
                raise ValueError(f"Empty input: {path.name}:{line}")
            yield {
                "id": f"wainject:{path.stem}:{row['id']}",
                "dataset": "wainject",
                "task": "prompt_injection",
                "input": state,
                "label": label,
                "stratum": f"{path.stem}:{label}",
                "group": digest(state),
                "source": str(path.relative_to(root)),
                "source_id": str(row["id"]),
                "line": line,
                "label_origin": "original benign/malicious directory",
                "family": path.stem,
            }


def agentharm_records(root, split="validation"):
    """Preserve original task groups and use only the released prompt as input."""
    if split not in ("validation", "test_public"):
        raise ValueError("Unknown original AgentHarm partition")
    for label_name, label in (("benign", 0), ("harmful", 1)):
        path = root / f"agentharm_{label_name}_behaviors_{split}.json"
        for row in json.loads(path.read_text())["behaviors"]:
            text = row["prompt"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("An original AgentHarm prompt is empty")
            yield {
                "id": f"agentharm:{split}:{label_name}:{row['id']}",
                "dataset": "agentharm", "task": "harmful_request", "input": text, "label": label,
                "split": "pilot_dev" if split == "validation" else "held_out_original",
                "original_split": split, "source": path.name, "source_id": str(row["id"]),
                "source_original_id": str(row["id_original"]),
                "group": f"agentharm:original:{row['id_original']}",
                "input_group": digest(text), "family": "original_agentharm_task",
                "label_origin": f"original {label_name} benchmark membership",
                "detailed_prompt": row["detailed_prompt"], "hint_included": row["hint_included"],
            }
