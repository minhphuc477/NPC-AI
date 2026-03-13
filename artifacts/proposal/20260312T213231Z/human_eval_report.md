# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5889 | (0.5537, 0.6278) |
| baseline_no_context | persona_consistency | 0.7796 | (0.7481, 0.8093) |
| baseline_no_context | naturalness | 0.7241 | (0.6963, 0.7519) |
| baseline_no_context | overall_quality | 0.6815 | (0.6537, 0.7111) |
| baseline_no_context_phi3_latest | context_relevance | 0.7611 | (0.7222, 0.8000) |
| baseline_no_context_phi3_latest | persona_consistency | 0.8167 | (0.7796, 0.8537) |
| baseline_no_context_phi3_latest | naturalness | 0.8037 | (0.7685, 0.8389) |
| baseline_no_context_phi3_latest | overall_quality | 0.7907 | (0.7574, 0.8223) |
| proposed_contextual_controlled | context_relevance | 0.7556 | (0.7315, 0.7796) |
| proposed_contextual_controlled | persona_consistency | 0.8407 | (0.8037, 0.8741) |
| proposed_contextual_controlled | naturalness | 0.7000 | (0.6685, 0.7315) |
| proposed_contextual_controlled | overall_quality | 0.7722 | (0.7519, 0.7926) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.6274 | 3 |
| persona_consistency | 0.5385 | 3 |
| naturalness | 0.5949 | 3 |
| overall_quality | 0.6188 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.6991 | 0.7945 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.4583 | 0.4286 |