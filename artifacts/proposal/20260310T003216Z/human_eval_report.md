# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5426 | (0.5093, 0.5778) |
| baseline_no_context | persona_consistency | 0.7815 | (0.7500, 0.8130) |
| baseline_no_context | naturalness | 0.7389 | (0.7093, 0.7667) |
| baseline_no_context | overall_quality | 0.6593 | (0.6259, 0.6907) |
| baseline_no_context_phi3_latest | context_relevance | 0.7352 | (0.6963, 0.7741) |
| baseline_no_context_phi3_latest | persona_consistency | 0.8537 | (0.8185, 0.8870) |
| baseline_no_context_phi3_latest | naturalness | 0.7981 | (0.7611, 0.8352) |
| baseline_no_context_phi3_latest | overall_quality | 0.7500 | (0.7185, 0.7815) |
| proposed_contextual_controlled | context_relevance | 0.7722 | (0.7519, 0.7908) |
| proposed_contextual_controlled | persona_consistency | 0.8815 | (0.8481, 0.9130) |
| proposed_contextual_controlled | naturalness | 0.6481 | (0.6203, 0.6796) |
| proposed_contextual_controlled | overall_quality | 0.7685 | (0.7481, 0.7907) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.6818 | 3 |
| persona_consistency | 0.4752 | 3 |
| naturalness | 0.5062 | 3 |
| overall_quality | 0.5805 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.7269 | 0.7882 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.5417 | 0.5634 |