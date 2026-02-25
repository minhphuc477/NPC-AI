# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5981 | (0.5593, 0.6370) |
| baseline_no_context | persona_consistency | 0.7796 | (0.7500, 0.8111) |
| baseline_no_context | naturalness | 0.7259 | (0.6907, 0.7593) |
| baseline_no_context | overall_quality | 0.6722 | (0.6463, 0.7000) |
| baseline_no_context_phi3_latest | context_relevance | 0.6519 | (0.6093, 0.6926) |
| baseline_no_context_phi3_latest | persona_consistency | 0.8333 | (0.8000, 0.8667) |
| baseline_no_context_phi3_latest | naturalness | 0.7944 | (0.7593, 0.8278) |
| baseline_no_context_phi3_latest | overall_quality | 0.7185 | (0.6907, 0.7482) |
| proposed_contextual_controlled | context_relevance | 0.8037 | (0.7833, 0.8259) |
| proposed_contextual_controlled | persona_consistency | 0.8722 | (0.8389, 0.9037) |
| proposed_contextual_controlled | naturalness | 0.6870 | (0.6630, 0.7130) |
| proposed_contextual_controlled | overall_quality | 0.7870 | (0.7667, 0.8074) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.4595 | 3 |
| persona_consistency | 0.4204 | 3 |
| naturalness | 0.3810 | 3 |
| overall_quality | 0.4424 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.7361 | 0.8493 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.6713 | 0.7403 |