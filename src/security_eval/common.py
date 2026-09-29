"""Small shared utilities. Secrets must never be serialized in run artifacts."""

import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def iter_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path):
    return list(iter_jsonl(path))


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values))


def key_from_file(path=None):
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    text = Path(path or ROOT / "openrouter.txt").read_text().strip()
    matches = re.findall(r"sk-or-[A-Za-z0-9_-]+", text)
    if len(set(matches)) != 1:
        raise ValueError("Credential file must contain exactly one OpenRouter key; contents omitted")
    return matches[0]


def shell_config_value(name, path=None):
    """Read one literal assignment without executing shell startup commands."""
    if name in os.environ:
        return os.environ[name]
    text = Path(path or Path.home() / ".zshrc").read_text()
    matches = re.findall(r"^\s*(?:export\s+)?" + re.escape(name) + r"\s*=\s*(.*?)\s*$", text, re.MULTILINE)
    if not matches:
        raise ValueError(f"Missing {name}; configuration contents omitted")
    tokens = shlex.split(matches[-1], comments=True)
    if len(tokens) != 1 or any(s in tokens[0] for s in ("$", "`", "\n")):
        raise ValueError(f"{name} must be a literal assignment; shell evaluation is disabled")
    return tokens[0]


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def code_digest():
    return digest({str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / "src").rglob("*.py"))})
