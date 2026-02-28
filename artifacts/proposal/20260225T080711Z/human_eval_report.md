# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5889 | (0.5500, 0.6296) |
| baseline_no_context | persona_consistency | 0.7537 | (0.7185, 0.7907) |
| baseline_no_context | naturalness | 0.7463 | (0.7148, 0.7759) |
| baseline_no_context | overall_quality | 0.6759 | (0.6444, 0.7093) |
| baseline_no_context_phi3_latest | context_relevance | 0.7259 | (0.6852, 0.7667) |
| baseline_no_context_phi3_latest | persona_consistency | 0.8315 | (0.8000, 0.8612) |
| baseline_no_context_phi3_latest | naturalness | 0.8185 | (0.7833, 0.8537) |
| baseline_no_context_phi3_latest | overall_quality | 0.7111 | (0.6833, 0.7407) |
| proposed_contextual_controlled | context_relevance | 0.7963 | (0.7722, 0.8222) |
| proposed_contextual_controlled | persona_consistency | 0.8981 | (0.8593, 0.9333) |
| proposed_contextual_controlled | naturalness | 0.7204 | (0.6907, 0.7519) |
| proposed_contextual_controlled | overall_quality | 0.7963 | (0.7759, 0.8185) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.5387 | 3 |
| persona_consistency | 0.5586 | 3 |
| naturalness | 0.5078 | 3 |
| overall_quality | 0.5697 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.7361 | 0.7865 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.6759 | 0.7568 |