# Evaluation data

## Benchmark sources

| Benchmark | Source | Evaluated revision |
| --- | --- | --- |
| R-Judge | [Official repository](https://github.com/Lordog/R-Judge) | `83ce301da3ad50dd8b397e772863f5411c3d3dc2` |
| WAInjectBench | [Official repository](https://github.com/Norrrrrrr-lyn/WAInjectBench) | `ee55236860820b3639c9039a2aa367b2cb86d114` |
| AgentHarm | [Official dataset](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) | `e23b3fe60a0da9037314b88e5ee3a0c054970dad` |

The pinned local R-Judge and WAInjectBench trees contain no top-level license file establishing a general redistribution grant. Their text is not mirrored in this package. Source availability alone is not treated as a blanket redistribution license.

AgentHarm's [dataset card](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) and [license](https://huggingface.co/datasets/ai-safety-institute/AgentHarm/blob/main/LICENSE) specify an MIT license with an additional restriction to improving AI safety and security. The dataset card requests evaluation-only use rather than training. The notice is reproduced below. This package also omits AgentHarm prompt text.

`partitions.jsonl.gz` records IDs, original binary labels, grouping metadata, and study roles. Original text can be obtained from the named upstream revision subject to its conditions. The exported metadata is not a substitute for the original benchmark license.

## Record fields and populations

These are model observations on existing public benchmarks, not a newly constructed or manually labeled security dataset. Labels are the original benchmark labels. All numeric unsafe scores lie in `[0, 1]`, and the positive class is task-specific.

| Field | Meaning |
| --- | --- |
| `id`, `dataset`, `partition` | Original benchmark identity and frozen study role. |
| `label` | Original binary label, with one denoting the task's unsafe class. |
| `group`, `family` | Recorded grouping units for clustered analysis. |
| `input_sha256` | Hash recorded for the formatted input. |
| `model`, `source_model`, `recovery_kind` | Analysis alias, actual source configuration, and recovery provenance. |
| `result.status`, `prediction`, `p_unsafe` | Execution outcome, decision, and usable class score. |
| `result.score_kind`, `score_status` | Probability readout and availability contract. |
| `result.usage`, `token_accounting` | Provider-reported token accounting when available. |
| `result.provider`, `returned_model`, `system_fingerprint` | Returned service identifiers when available. |
| `result.response_sha256`, `request_folder` | Link to the immutable local request/response archive. |

Only an explicit field allowlist is exported to the prediction files. Original benchmark input text, interaction transcripts, model reasoning, raw HTTP payloads, headers, credentials, account identifiers, and personal filesystem paths are excluded. Full datasets and trajectories are preserved locally.

Selected private service identifiers use stable publication aliases, and private endpoint URLs are withheld. Configuration and recovery cohorts remain distinct. Labels, predictions, probabilities, token counts, and statistical results are unchanged.

Classification-only configurations can have null scores. Every final probability and policy observation has a finite score; the final configurations cover all applicable inputs. Changed-format responses retain their score type in the final probability data. Intermediate attempts and retry candidate histories are not distributed.

WAInjectBench uses known source families for clustered uncertainty, R-Judge uses exact-input groups, and AgentHarm uses original task groups shared across variants. The prediction row count exceeds the number of unique benchmark inputs because models and followup cohorts reuse the same inputs.

Frozen partitions contain 2,200 test inputs: 1,612 WAInjectBench-text, 236 R-Judge, and 352 AgentHarm. AgentHarm has no independent selection or confirmation split in this study. No selection or confirmation records are included for AgentHarm.

The package is intended for research on AI safety and security. Consult the original data sources and their conditions before obtaining benchmark text or conducting new inference. This artifact does not grant new rights over the underlying benchmarks or model services.

<details>
<summary>AgentHarm license notice</summary>

```text
MIT License with an additional clause

Copyright (c) 2024 Gray Swan AI and UK AI Safety Institute

We prohibit using the dataset and benchmark for purposes besides improving the
safety and security of AI systems.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>
