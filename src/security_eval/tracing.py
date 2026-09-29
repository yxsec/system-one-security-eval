"""Durable, linked research traces with credential redaction and explicit provenance."""

import contextvars
import functools
import gzip
import hashlib
import json
import os
import platform
import time
import traceback
import uuid
import warnings
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

from .common import ROOT, write_json

ACTIVE = contextvars.ContextVar("research_trace", default=None)
SECRET_FIELDS = {"authorization", "proxy-authorization", "cookie", "set-cookie", "api_key",
                 "apikey", "access_token", "refresh_token", "password", "client_secret"}


def json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (datetime, Path)):
        return str(value)
    raise TypeError(f"Unsupported trace value: {type(value).__name__}")


class TraceRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self.secrets = set()
        self.io_seconds = 0.0

    def redact(self, value):
        if isinstance(value, dict):
            return {k: "[REDACTED]" if str(k).lower() in SECRET_FIELDS else self.redact(v)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, str):
            for secret in sorted(self.secrets, key=len, reverse=True):
                if secret:
                    value = value.replace(secret, "[REDACTED]")
        return value

    def emit(self, event, **data):
        start = time.perf_counter()
        active = ACTIVE.get()
        record = {"schema_version": 1, "event_id": uuid.uuid4().hex,
                  "session_id": self.session_id, "pid": os.getpid(),
                  "timestamp_utc": datetime.now(UTC).isoformat(), "monotonic_ns": time.monotonic_ns(),
                  "span_id": active[1] if active and active[0] is self else None,
                  "event": event, **data}
        serializable = json.loads(json.dumps(record, default=json_default, ensure_ascii=False, allow_nan=False))
        record = self.redact(serializable)
        content = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
        if self.path.suffix == ".gz":
            content = gzip.compress(content, compresslevel=3, mtime=0)
        # Persist each event before the next external operation can begin.
        with self.path.open("ab") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        self.io_seconds += time.perf_counter() - start
        return record["event_id"]

    @contextmanager
    def span(self, kind, **data):
        parent = ACTIVE.get()
        span_id = uuid.uuid4().hex
        token = ACTIVE.set((self, span_id))
        try:
            self.emit("span_start", kind=kind, parent_span_id=parent[1] if parent else None, **data)
            yield span_id
        except BaseException as exc:
            self.emit("span_error", kind=kind, error_type=type(exc).__name__, error_message=str(exc),
                      traceback=traceback.format_exc())
            raise
        else:
            self.emit("span_end", kind=kind)
        finally:
            ACTIVE.reset(token)


def emit(event, **data):
    active = ACTIVE.get()
    if active:
        return active[0].emit(event, **data)
    return None


@contextmanager
def capture_warnings():
    with warnings.catch_warnings(record=True) as caught:
        try:
            yield
        finally:
            if caught:
                emit("python_warnings", warnings=[{
                    "category": warning.category.__name__, "message": str(warning.message),
                    "filename": warning.filename, "line": warning.lineno,
                } for warning in caught])


def measured_start():
    active = ACTIVE.get()
    return time.perf_counter(), active[0].io_seconds if active else 0.0


def elapsed_ms(start):
    active = ACTIVE.get()
    logging_seconds = (active[0].io_seconds if active else 0.0) - start[1]
    return (time.perf_counter() - start[0] - logging_seconds) * 1000


def token_accounting(result):
    usage = result.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    return {"provider_or_engine_usage": usage,
            "reported_input_tokens": input_tokens, "reported_output_tokens": output_tokens,
            "reported_total_tokens": usage.get("total_tokens"),
            "derived_total_tokens": input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None else None,
            "local_preflight_input_tokens": result.get("input_tokens"),
            "local_padded_input_tokens": result.get("padded_input_tokens"),
            "missing_usage_is_not_zero": True}


def recorded_decision(function):
    @functools.wraps(function)
    def wrapped(self, task, text):
        active = ACTIVE.get()
        recorder = active[0] if active else TraceRecorder(ROOT / "runs/standalone/trajectory.jsonl")
        wall_start, io_start = time.perf_counter(), recorder.io_seconds
        client = getattr(self, "client", None)
        if client:
            authorization = client.headers.get("authorization", "")
            recorder.secrets.update([authorization, authorization.removeprefix("Bearer ")])
        with recorder.span("decision", task=task, input=text,
                           model_metadata=getattr(self, "metadata", {})) as span_id:
            start = measured_start()
            with capture_warnings():
                result = function(self, task, text)
            result["trace_id"] = span_id
            result["token_accounting"] = token_accounting(result)
            result["decision_elapsed_excluding_trace_io_ms"] = elapsed_ms(start)
            recorder.emit("decision_result", result=result)
        result["decision_wall_ms_including_trace_io"] = (time.perf_counter() - wall_start) * 1000
        result["trace_io_ms"] = (recorder.io_seconds - io_start) * 1000
        return result
    return wrapped


def post_json(client, url, payload):
    active = ACTIVE.get()
    recorder = active[0] if active else TraceRecorder(ROOT / "runs/standalone/trajectory.jsonl")
    request = client.build_request("POST", url, json=payload)
    authorization = request.headers.get("authorization", "")
    recorder.secrets.update([authorization, authorization.removeprefix("Bearer ")])
    with recorder.span("http_attempt", method=request.method, url=str(request.url),
                       headers=dict(request.headers), request_body=payload,
                       request_body_text=request.content.decode("utf-8")):
        start = time.perf_counter()
        response = client.send(request)
        network_ms = (time.perf_counter() - start) * 1000
        try:
            body = response.json()
        except ValueError:
            body = None
        recorder.emit("http_response", status_code=response.status_code, headers=dict(response.headers),
                      response_body=body, response_body_text=response.text,
                      response_bytes_sha256=hashlib.sha256(response.content).hexdigest(),
                      response_byte_count=len(response.content), network_latency_ms=network_ms,
                      usage=body.get("usage") if isinstance(body, dict) else None)
        return response


def snapshot_environment(folder):
    folder = Path(folder)
    if (folder / "environment.json").exists():
        return
    report = {"captured_utc": datetime.now(UTC).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "machine": platform.machine(),
              "cpu_count": os.cpu_count(), "packages": sorted(
                  [{"name": d.metadata["Name"], "version": d.version} for d in distributions()],
                  key=lambda row: row["name"].lower()),
              "network_region": "not independently measured", "environment_variables": "not dumped"}
    hardware = ROOT / "reports/hardware_probe.json"
    if hardware.exists():
        report["earlier_hardware_probe"] = json.loads(hardware.read_text())
    write_json(folder / "environment.json", report)
    write_json(folder / "reproduction_snapshot.json", {
        str(path.relative_to(ROOT)): path.read_text()
        for path in [ROOT / "uv.lock", ROOT / "pyproject.toml", ROOT / "protocol/source_lock.json",
                     ROOT / "protocol/model_lock.json"] if path.exists()
    })


def seal_artifacts(folder):
    folder = Path(folder)
    files = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name != "artifact_checksums.json":
            with path.open("rb") as stream:
                files[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    write_json(folder / "artifact_checksums.json", {
        "algorithm": "sha256", "files": files,
    })
