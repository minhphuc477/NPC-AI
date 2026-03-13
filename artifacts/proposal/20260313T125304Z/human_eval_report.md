# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5741 | (0.5426, 0.6074) |
| baseline_no_context | persona_consistency | 0.7648 | (0.7333, 0.7963) |
| baseline_no_context | naturalness | 0.7222 | (0.6963, 0.7500) |
| baseline_no_context | overall_quality | 0.6481 | (0.6204, 0.6778) |
| baseline_no_context_phi3_latest | context_relevance | 0.6704 | (0.6296, 0.7130) |
| baseline_no_context_phi3_latest | persona_consistency | 0.8407 | (0.8037, 0.8723) |
| baseline_no_context_phi3_latest | naturalness | 0.7648 | (0.7296, 0.8019) |
| baseline_no_context_phi3_latest | overall_quality | 0.7111 | (0.6833, 0.7407) |
| proposed_contextual_controlled_tuned | context_relevance | 0.8259 | (0.8056, 0.8463) |
| proposed_contextual_controlled_tuned | persona_consistency | 0.8944 | (0.8630, 0.9222) |
| proposed_contextual_controlled_tuned | naturalness | 0.7407 | (0.7056, 0.7759) |
| proposed_contextual_controlled_tuned | overall_quality | 0.8111 | (0.7870, 0.8352) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.7582 | 3 |
| persona_consistency | 0.5415 | 3 |
| naturalness | 0.5176 | 3 |
| overall_quality | 0.6147 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_tuned_vs_baseline_no_context | 0.8102 | 0.8941 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | 0.6898 | 0.7733 |