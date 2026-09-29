"""Pinned upstream decision scorers with explicit context and calibration contracts."""

import hashlib
import json
import sys
from importlib.metadata import version

from .adapters import CHOICES, QUESTIONS, choice_probability, typed_question
from .common import ROOT
from .tracing import elapsed_ms, emit, measured_start, recorded_decision


def length_failure(n, limit, start):
    return {"status": "unsupported_length", "input_tokens": n, "max_input_tokens": limit,
            "latency_ms": elapsed_ms(start), "cost_usd": None}


class DeciderAdapter:
    def __init__(self, config):
        import torch

        sys.path.insert(0, str(ROOT / "external/decider"))
        from decider.infer import Decider

        path = ROOT / config["local_path"]
        shipped = json.loads((path / "decider_config.json").read_text())
        torch.set_num_threads(4)
        self.torch = torch
        self.agent = Decider(str(path), device=config["device"], use_graphs=False)
        self.parent_control = config.get("parent_control", False)
        self.candidate_order = config.get("candidate_order", ["A", "B"])
        self.limit = shipped["max_state_tokens"]
        self.metadata = {**config, "device": str(self.agent.dev), "dtype": str(self.agent.m.dtype)
                         if hasattr(self.agent.m, "dtype") else str(next(self.agent.m.parameters()).dtype),
                         "shipped_config": shipped, "max_input_tokens": self.limit,
                         "temperature": self.agent.T, "layout": "state_first", "batch_size": 1,
                         "torch": torch.__version__, "transformers": version("transformers")}

    @recorded_decision
    def predict(self, task, text):
        start = measured_start()
        question = {"risk": typed_question(task, self.candidate_order)}
        # Inspect the exact upstream prompt before its context-cropping path can run.
        rendered, index, items = self.agent._system_one_items(text, question, max_state_tokens=1_000_000)
        emit("local_prepared_input", questions=question, rendered_questions=rendered,
             answer_index=index, items=items, truncation=False, padding_multiple=64)
        n = max(len(item["ids"]) for item in items)
        padded_n = ((n + 63) // 64) * 64
        if padded_n > self.limit:
            return length_failure(n, self.limit, start)
        result = self.agent.system_one(text, question, max_state_tokens=self.limit)
        if str(self.agent.dev).startswith("mps"):
            self.torch.mps.synchronize()
        emit("local_native_response", response=result)
        answer = result["answers"]["risk"]
        prob = choice_probability(answer)
        return {"status": "ok", "prediction": int(prob >= 0.5), "p_unsafe": prob,
                "score_kind": "candidate_token_conditional_uncalibrated_parent" if self.parent_control
                else "native_class_probability_shipped_temperature",
                "latency_ms": elapsed_ms(start), "input_tokens": n,
                "padded_input_tokens": padded_n, "cost_usd": None, "typed_answer": answer,
                "usage": result.get("usage", {})}


class NimbleAdapter:
    def __init__(self, config):
        self.parent_control = False
        self.candidate_order = config.get("candidate_order", ["A", "B"])
        sys.path.insert(0, str(ROOT / "external/nimble"))
        from nimble.scoring.calibration import ADAPTER_REVISIONS, served_temperature
        from nimble.scoring.parallel_scorer import ParallelScorer

        path = ROOT / config["local_path"]
        ready = json.loads((path / "READY.json").read_text())
        shipped = json.loads((path / "schema_config.json").read_text())
        prompt_file = ROOT / "external/nimble/nimble/scoring/parallel_schema.py"
        if hashlib.sha256(prompt_file.read_bytes()).hexdigest() != shipped["prompt_code_sha256"]:
            raise ValueError("Nimble scoring template does not match the published adapter contract")
        if shipped["task"] != "schema_candidate_classification_v1":
            raise ValueError("Unexpected Nimble adapter task")
        temperature = served_temperature(ready)
        calibration_revision = ADAPTER_REVISIONS.get(ready["adapter_sha256"])
        self.limit = int(config.get("max_input_tokens", shipped.get("max_length", 2048)))
        self.scorer = ParallelScorer(model_path=str(path), model_id=config["repo"],
                                     revision=config["revision"], max_input_tokens=self.limit,
                                     temperature=temperature)
        self.metadata = {**config, "device": "mlx_metal", "dtype": str(self.scorer.head_weight.dtype),
                         "max_input_tokens": self.scorer.max_input_tokens,
                         "temperature": temperature, "calibration_revision_by_adapter_hash": calibration_revision,
                         "calibration_provenance": "upstream served_temperature using verified adapter hash",
                         "merge_manifest": ready, "batch_size": 1, "mlx": version("mlx"),
                         "mlx_lm": version("mlx-lm"), "transformers": version("transformers")}

    @recorded_decision
    def predict(self, task, text):
        from nimble.scoring.parallel_schema import prepare_prompts

        start = measured_start()
        schema = {"risk": {"type": "enum", "description": QUESTIONS[task],
                           "choices": self.candidate_order, "choice_descriptions": CHOICES}}
        prepared = prepare_prompts(self.scorer.tokenizer, text, schema, 1_000_000,
                                   system_role=self.scorer.system_role)
        emit("local_prepared_input", schema=schema, full_input_ids=prepared.full_ids,
             prefix_ids=prepared.prefix_ids, suffix_ids=prepared.suffix_ids,
             candidate_ids=prepared.candidate_ids, truncation=False)
        n = max(map(len, prepared.full_ids))
        if n > self.scorer.max_input_tokens:
            return length_failure(n, self.scorer.max_input_tokens, start)
        result = self.scorer.score(text, schema)
        emit("local_native_response", response=result, generated_output_tokens=0)
        field = result["fields"]["risk"]
        answer = {"choice": field["value"], "probabilities": field["scores"]}
        prob = choice_probability(answer)
        return {"status": "ok", "prediction": int(prob >= 0.5), "p_unsafe": prob,
                "score_kind": "candidate_token_conditional_uncalibrated_parent" if self.parent_control
                else "native_class_probability_shipped_temperature",
                "latency_ms": elapsed_ms(start), "input_tokens": n,
                "usage": {"input_tokens": sum(map(len, prepared.full_ids)), "output_tokens": 0},
                "cost_usd": None, "typed_answer": answer, "candidate_logits": field["logits"],
                "engine_metrics": result["metrics"]}


class NimbleParentAdapter(NimbleAdapter):
    """Unadapted pinned base with the identical candidate scorer and temperature one."""

    def __init__(self, config):
        sys.path.insert(0, str(ROOT / "external/nimble"))
        from nimble.scoring.parallel_scorer import ParallelScorer

        self.parent_control = True
        self.candidate_order = config.get("candidate_order", ["A", "B"])
        self.scorer = ParallelScorer(model_path=str(ROOT / config["local_path"]),
                                     model_id=config["repo"], revision=config["revision"],
                                     max_input_tokens=int(config.get("max_input_tokens", 2048)), temperature=1.0, allow_uncalibrated=True)
        self.metadata = {**config, "device": "mlx_metal", "dtype": str(self.scorer.head_weight.dtype),
                         "max_input_tokens": self.scorer.max_input_tokens, "temperature": 1.0,
                         "calibration_provenance": "none; unadapted parent control",
                         "batch_size": 1, "mlx": version("mlx"), "mlx_lm": version("mlx-lm")}
