# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5815 | (0.5463, 0.6185) |
| baseline_no_context | persona_consistency | 0.7981 | (0.7648, 0.8315) |
| baseline_no_context | naturalness | 0.7593 | (0.7278, 0.7907) |
| baseline_no_context | overall_quality | 0.6833 | (0.6593, 0.7093) |
| baseline_no_context_phi3_latest | context_relevance | 0.7352 | (0.6981, 0.7741) |
| baseline_no_context_phi3_latest | persona_consistency | 0.8500 | (0.8167, 0.8833) |
| baseline_no_context_phi3_latest | naturalness | 0.8407 | (0.8074, 0.8741) |
| baseline_no_context_phi3_latest | overall_quality | 0.7648 | (0.7370, 0.7926) |
| proposed_contextual_controlled | context_relevance | 0.7981 | (0.7815, 0.8149) |
| proposed_contextual_controlled | persona_consistency | 0.8611 | (0.8259, 0.8963) |
| proposed_contextual_controlled | naturalness | 0.7056 | (0.6685, 0.7389) |
| proposed_contextual_controlled | overall_quality | 0.7778 | (0.7593, 0.7963) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.4925 | 3 |
| persona_consistency | 0.3523 | 3 |
| naturalness | 0.4346 | 3 |
| overall_quality | 0.3566 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.7222 | 0.8243 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.5231 | 0.5439 |