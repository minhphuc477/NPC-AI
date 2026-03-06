# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5926 | (0.5556, 0.6296) |
| baseline_no_context | persona_consistency | 0.7500 | (0.7148, 0.7833) |
| baseline_no_context | naturalness | 0.7519 | (0.7204, 0.7833) |
| baseline_no_context | overall_quality | 0.6944 | (0.6666, 0.7259) |
| baseline_no_context_phi3_latest | context_relevance | 0.6759 | (0.6315, 0.7185) |
| baseline_no_context_phi3_latest | persona_consistency | 0.7704 | (0.7389, 0.8037) |
| baseline_no_context_phi3_latest | naturalness | 0.7870 | (0.7500, 0.8241) |
| baseline_no_context_phi3_latest | overall_quality | 0.7278 | (0.6981, 0.7593) |
| proposed_contextual_controlled | context_relevance | 0.7926 | (0.7740, 0.8111) |
| proposed_contextual_controlled | persona_consistency | 0.9259 | (0.8963, 0.9519) |
| proposed_contextual_controlled | naturalness | 0.7019 | (0.6722, 0.7315) |
| proposed_contextual_controlled | overall_quality | 0.8037 | (0.7851, 0.8241) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.6404 | 3 |
| persona_consistency | 0.5271 | 3 |
| naturalness | 0.4415 | 3 |
| overall_quality | 0.5225 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.7315 | 0.8378 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.6806 | 0.7746 |