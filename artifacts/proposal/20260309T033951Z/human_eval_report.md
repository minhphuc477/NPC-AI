# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5574 | (0.5204, 0.5944) |
| baseline_no_context | persona_consistency | 0.7870 | (0.7574, 0.8185) |
| baseline_no_context | naturalness | 0.7593 | (0.7296, 0.7907) |
| baseline_no_context | overall_quality | 0.7185 | (0.6889, 0.7481) |
| baseline_no_context_phi3_latest | context_relevance | 0.7519 | (0.7074, 0.7945) |
| baseline_no_context_phi3_latest | persona_consistency | 0.8407 | (0.8055, 0.8741) |
| baseline_no_context_phi3_latest | naturalness | 0.8167 | (0.7833, 0.8519) |
| baseline_no_context_phi3_latest | overall_quality | 0.7574 | (0.7278, 0.7870) |
| proposed_contextual_controlled | context_relevance | 0.7852 | (0.7741, 0.7963) |
| proposed_contextual_controlled | persona_consistency | 0.8870 | (0.8500, 0.9204) |
| proposed_contextual_controlled | naturalness | 0.6204 | (0.5963, 0.6463) |
| proposed_contextual_controlled | overall_quality | 0.7648 | (0.7463, 0.7833) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.6801 | 3 |
| persona_consistency | 0.5460 | 3 |
| naturalness | 0.5087 | 3 |
| overall_quality | 0.5047 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.6065 | 0.6456 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.5185 | 0.5312 |