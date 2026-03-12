# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5093 | (0.4796, 0.5426) |
| baseline_no_context | persona_consistency | 0.7759 | (0.7444, 0.8074) |
| baseline_no_context | naturalness | 0.7185 | (0.6907, 0.7463) |
| baseline_no_context | overall_quality | 0.6630 | (0.6333, 0.6963) |
| baseline_no_context_phi3_latest | context_relevance | 0.7167 | (0.6741, 0.7574) |
| baseline_no_context_phi3_latest | persona_consistency | 0.7963 | (0.7611, 0.8315) |
| baseline_no_context_phi3_latest | naturalness | 0.7963 | (0.7611, 0.8296) |
| baseline_no_context_phi3_latest | overall_quality | 0.7611 | (0.7315, 0.7926) |
| proposed_contextual_controlled | context_relevance | 0.7778 | (0.7556, 0.7982) |
| proposed_contextual_controlled | persona_consistency | 0.8500 | (0.8111, 0.8852) |
| proposed_contextual_controlled | naturalness | 0.6944 | (0.6648, 0.7259) |
| proposed_contextual_controlled | overall_quality | 0.7741 | (0.7556, 0.7926) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.6877 | 3 |
| persona_consistency | 0.5071 | 3 |
| naturalness | 0.4540 | 3 |
| overall_quality | 0.5217 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.7130 | 0.8108 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.5324 | 0.5455 |