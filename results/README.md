# Final result populations

Each data file serves a specified final analysis. They are not successive checkpoints to concatenate.

| Prediction file in `data/predictions/` | Final analysis |
| --- | --- |
| `laya_all_partitions.jsonl.gz` | Complete Laya source and test observations for reproducing calibration and linking all main analyses. |
| `primary_classification.jsonl.gz` | The twelve primary configurations, including valid-input rankings and paired comparisons. |
| `additional_judges.jsonl.gz` | Final DeepSeek, GLM-5.3, Kimi, and Qwen3.8 test judgments. |
| `glm_completed.jsonl.gz` | Completed GLM-5.3-Flash judgments used in expanded rankings and paired-error analysis. |
| `completed_probabilities.jsonl.gz` | Final selected GPT-4.1, Qwen3-8B, and GLM-5.3-Flash responses for probability comparisons. |
| `frozen_policy_judges.jsonl.gz` | Completed GPT-4.1 and Qwen3-8B test scores used with parameters fitted under the same final scoring rule. The filename is retained for compatibility. |
| `threshold_sensitivity.jsonl.gz` | Source and test scores for comparing symmetric and independent thresholds. Test responses repeat the corresponding final scoring observations. |

`final_scoring_all_partitions.jsonl.gz` contains 25,842 observations for the six policy models across all 4,307 applicable inputs. `nimble_all_partitions.jsonl.gz` additionally retains both Nimble and its base-model control (8,614 observations). The completed scoring rule is applied across all partitions. Temperatures are refitted only on development and thresholds selected only on selection, then fixed for confirmation and test. The initial test results had already been observed, so this is a post-hoc reanalysis rather than a prospective evaluation. Classification-only configurations retain their distinct response contracts.

## Numerical results

- `heldout_current.json`: primary classification, shared-input populations, uncertainty, and fixed-policy outcomes.
- `added_models.json`: final expanded rankings, calibration, grouped intervals, and paired contrasts.
- `completed_probability.json`: final probability results on native and shared populations, including the sensitivity excluding JSON-format responses.
- `probability_followup.json`: final completed-score calibration and policy analysis for the two primary judges.
- `confidence_diagnostics.json`, `score_tie_diagnostics.json`: confidence, errors, and tied-score groups.
- `calibration_sensitivity.json`: alternative ECE bins and the empirical-prevalence Brier reference.
- `error_complementarity.json`, `error_transitions.csv`: corrected, repeated, and introduced errors across model pairs.
- `policy_budget_resolution.json`: error counts permitted by each limit and partition size.
- `threshold_policies.json`: 144 selection-only policies, with 72 corresponding symmetric policies reproduced exactly.
- `threshold_sensitivity.json`, `threshold_sensitivity.csv`: symmetric versus independent threshold results on selection, confirmation, and test partitions, with paired fixed-policy coverage intervals.
- `interpretation_sensitivity.json`, `T14_source_sensitivity.json`: attack-group and source-family results.
- `T12_paired_effects.json`, `T15_parent_controls.json`: paired model comparisons and adapted-model comparisons with their parent controls.

Within the final probability comparison, each model is evaluated on the same 1,612 WAInjectBench inputs, 236 R-Judge records, and 352 AgentHarm requests. Every applicable input is covered; the model-specific and common populations therefore coincide.

Laya uses the pinned English checkpoint with a 2,048-token formatted-input budget. Nimble and its base-model control use the scorer's 262,143-token acceptance cap; the maximum observed input is 2,155 tokens. This verifies input coverage for the benchmark, not classification quality near the architectural limit. Original restricted-input observations are archived locally.

The completed probability file contains 6,600 responses, or 2,200 per judge. A Qwen3-8B response uses the official local checkpoint after its hosted endpoint rejected that input. The same completed record is used by the final policy analysis, with source parameters fitted under the final scoring rule. Its label and probability come from the same forward pass. Recovery provenance is retained in the observations and `protocol/final_scoring.json`.
