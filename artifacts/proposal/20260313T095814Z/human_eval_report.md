# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.6130 | (0.5796, 0.6463) |
| baseline_no_context | persona_consistency | 0.7556 | (0.7241, 0.7870) |
| baseline_no_context | naturalness | 0.7148 | (0.6907, 0.7407) |
| baseline_no_context | overall_quality | 0.6944 | (0.6685, 0.7204) |
| baseline_no_context_phi3_latest | context_relevance | 0.6870 | (0.6463, 0.7260) |
| baseline_no_context_phi3_latest | persona_consistency | 0.7981 | (0.7630, 0.8333) |
| baseline_no_context_phi3_latest | naturalness | 0.7593 | (0.7222, 0.7963) |
| baseline_no_context_phi3_latest | overall_quality | 0.7019 | (0.6704, 0.7352) |
| proposed_contextual_controlled_tuned | context_relevance | 0.7889 | (0.7630, 0.8148) |
| proposed_contextual_controlled_tuned | persona_consistency | 0.8407 | (0.8056, 0.8759) |
| proposed_contextual_controlled_tuned | naturalness | 0.7315 | (0.6963, 0.7667) |
| proposed_contextual_controlled_tuned | overall_quality | 0.8222 | (0.7926, 0.8500) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.5541 | 3 |
| persona_consistency | 0.4949 | 3 |
| naturalness | 0.5352 | 3 |
| overall_quality | 0.5607 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_tuned_vs_baseline_no_context | 0.7639 | 0.9014 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | 0.6944 | 0.8000 |