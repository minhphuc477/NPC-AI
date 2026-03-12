# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.6185 | (0.5759, 0.6630) |
| baseline_no_context | persona_consistency | 0.8019 | (0.7704, 0.8333) |
| baseline_no_context | naturalness | 0.7222 | (0.6907, 0.7519) |
| baseline_no_context | overall_quality | 0.6667 | (0.6407, 0.6963) |
| baseline_no_context_phi3_latest | context_relevance | 0.7296 | (0.6926, 0.7648) |
| baseline_no_context_phi3_latest | persona_consistency | 0.8611 | (0.8296, 0.8926) |
| baseline_no_context_phi3_latest | naturalness | 0.8352 | (0.8000, 0.8685) |
| baseline_no_context_phi3_latest | overall_quality | 0.7407 | (0.7111, 0.7723) |
| proposed_contextual_controlled | context_relevance | 0.7981 | (0.7796, 0.8148) |
| proposed_contextual_controlled | persona_consistency | 0.8611 | (0.8278, 0.8926) |
| proposed_contextual_controlled | naturalness | 0.6981 | (0.6648, 0.7315) |
| proposed_contextual_controlled | overall_quality | 0.7741 | (0.7500, 0.7981) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.5065 | 3 |
| persona_consistency | 0.4239 | 3 |
| naturalness | 0.4111 | 3 |
| overall_quality | 0.4102 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.7269 | 0.8101 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.5602 | 0.6032 |