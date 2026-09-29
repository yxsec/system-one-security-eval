# System One Security Evaluation

Final experimental data, analysis code, and results for evaluating System One models on prompt-injection detection, interaction-risk judgment, and harmful-request admission.

## Contents

| Directory | Contents |
| --- | --- |
| `data/predictions/` | Final response-level labels, probabilities, input IDs, and token accounting used in the reported analyses. |
| `data/` | Benchmark sources, label definitions, and fixed partition membership. |
| `protocol/` | Model revisions, analysis settings, calibration parameters, and selected thresholds. |
| `analysis/` | Code for rankings, calibration, paired comparisons, and error complementarity. |
| `src/` | Experiment runner, model adapters, probability extraction, tracing, and metrics. |
| `scripts/` | Pinned source and model setup, benchmark input reconstruction, and run summaries. |
| `results/` | Final numerical summaries and tables. |

The study evaluates 2,200 held-out inputs: 1,612 WAInjectBench, 236 R-Judge, and 352 AgentHarm inputs. It includes twelve primary configurations and five additional judges. All labels come from the original benchmarks. The main Laya configuration uses a 2,048-token formatted-input budget and covers all 2,200 test inputs. The pinned Nimble and base-model configurations also cover every input, with a scorer acceptance cap of 262,143 tokens and an observed maximum of 2,155 formatted tokens. See [result definitions](results/README.md) for the exact data used in each analysis.

This repository excludes intermediate attempts, retry candidates, failure diagnostics, historical reports, manuscript sources, PDFs, and image-generation assets. Benchmark input text and raw responses are not redistributed. Original input text can be obtained from the pinned [benchmark sources](data/README.md#benchmark-sources).

## Reproduce the final results

Python 3.11 or newer can verify file hashes and reproduce the point estimates without additional packages:

```bash
python3 reproduce_results.py
```

For grouped confidence intervals and the additional analyses:

```bash
python3 -m pip install -r requirements-analysis.txt
python3 reproduce_intervals.py
python3 analysis/error_complementarity/verify.py
python3 analysis/calibration_sensitivity/analyze.py --check
python3 analysis/completed_probabilities/analyze.py --check
python3 analysis/laya_configuration/verify.py
python3 analysis/threshold_sensitivity/analyze.py --check
python3 analysis/threshold_sensitivity/verify.py
```

These commands use the distributed observations and make no model or network requests. They require no credentials or GPU.

The threshold sensitivity analysis compares symmetric and independently selected allowance/blocking thresholds on the same saved scores. It selects each policy on source inputs and evaluates it unchanged on confirmation and test inputs. `results/threshold_sensitivity.csv` contains all 144 policies, with action counts and error-budget checks. The accompanying data includes the completed source observations needed to reproduce selection and the matching final test responses. Temperatures are refitted on development only and thresholds selected on selection only. This completion and reanalysis followed initial test inspection; it is not a new prospective validation. The policy comparison covers WAInjectBench and R-Judge, which have separate selection and confirmation partitions. AgentHarm has no such partitions in this protocol and remains in the classification, calibration, and error-complementarity analyses.

Regenerate its action-coverage figure with `python3 analysis/threshold_sensitivity/plot.py --output results/regenerated_threshold_sensitivity`. This writes PDF, SVG, and PNG files without changing the final numerical summaries.

## Run new experiments

The core runner retains the study's task prompts, local model adapters, final-answer probability extraction, and credential-redacted request/response recording. It runs one request per input, with no historical retry loops or automatic provider changes. New observations go into a separate directory and do not overwrite the distributed results.

Install the package and restore the pinned benchmark and model implementation sources:

```bash
python3 -m pip install -e .
python3 scripts/bootstrap_sources.py
python3 scripts/fetch_agentharm.py
python3 scripts/prepare_inputs.py
```

For hosted models, configure the environment variables listed in `protocol/models.json`. OpenRouter uses `OPENROUTER_API_KEY`, and the native Qwen probability configuration uses `DASHSCOPE_API_KEY`. The generic hosted-judge configurations use `HOSTED_JUDGE_BASE_URL` and `HOSTED_JUDGE_API_KEY`. Provide a compatible endpoint for those configurations. No credentials or private endpoint addresses are distributed.

Validate the inputs without loading a model or making requests, then run an experiment:

```bash
security-eval --model jev --dataset wainject --partition test --output runs/jev_wainject --dry-run
security-eval --model jev --dataset wainject --partition test --output runs/jev_wainject
python3 scripts/summarize_run.py --predictions runs/jev_wainject/predictions.jsonl --output runs/jev_wainject/metrics.json
```

`protocol/models.json` includes the twelve primary configurations, five additional judges, and separate GPT/Qwen probability configurations. Change `--model`, `--dataset`, or `--partition` to run another supported combination. Use `--limit` for a small check. Hosted service availability and model routing can change, so a new run need not reproduce the original responses exactly.

Local models additionally need their inference dependencies and pinned weights:

```bash
python3 -m pip install -e '.[local]'
python3 scripts/prepare_models.py --models laya protectai decider piguard prompt_guard_2 wildguard
```

The original Decider and WildGuard configurations use Apple MPS. Nimble uses Apple MLX and its upstream adapter merge. For that configuration, install the `apple` extra, download Nimble with `scripts/prepare_models.py --models nimble`, and run `scripts/merge_nimble.py`. The Decider parent additionally needs `scripts/prepare_models.py --models decider_parent` and `scripts/prepare_decider_parent.py`. Model access restrictions and upstream dependencies still apply.

The probability-readout regression tests run with `python3 -m pytest tests` after installing pytest.

To regenerate the expanded rankings or paired-error analysis:

```bash
python3 analysis/additional_judges/analyze.py
python3 analysis/error_complementarity/analyze.py
```

The regenerated summaries are written to `results/regenerated_*.json`, preserving the distributed final summaries.

## Data conventions

Each prediction file identifies its analysis role. Classification, completed probability comparisons, and frozen-policy evaluation use the response cohorts specified in the protocol. A label and its probability always come from the same response. Do not concatenate files and count repeated IDs as independent benchmark inputs.

For repeated requests, the retained response is selected by output validity without consulting the benchmark label. Changed output formats remain identifiable in the final records. Model identity, probability type, and response hashes are retained where available. Selected private service identifiers use stable aliases without changing numeric observations.

`manifest.json` hashes every distributed file. See [data documentation](data/README.md#record-fields-and-populations) for field definitions.

Benchmark sources and the AgentHarm license notice are documented in [data/README.md](data/README.md).
