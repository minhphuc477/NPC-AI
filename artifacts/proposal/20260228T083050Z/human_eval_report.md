# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5519 | (0.5185, 0.5852) |
| baseline_no_context | persona_consistency | 0.7407 | (0.7074, 0.7741) |
| baseline_no_context | naturalness | 0.7241 | (0.6944, 0.7519) |
| baseline_no_context | overall_quality | 0.6759 | (0.6500, 0.7056) |
| baseline_no_context_phi3_latest | context_relevance | 0.7093 | (0.6704, 0.7481) |
| baseline_no_context_phi3_latest | persona_consistency | 0.8870 | (0.8574, 0.9167) |
| baseline_no_context_phi3_latest | naturalness | 0.8444 | (0.8111, 0.8778) |
| baseline_no_context_phi3_latest | overall_quality | 0.7593 | (0.7278, 0.7907) |
| proposed_contextual_controlled | context_relevance | 0.8648 | (0.8444, 0.8852) |
| proposed_contextual_controlled | persona_consistency | 0.9204 | (0.8944, 0.9444) |
| proposed_contextual_controlled | naturalness | 0.7296 | (0.6981, 0.7611) |
| proposed_contextual_controlled | overall_quality | 0.8296 | (0.8111, 0.8463) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.6422 | 3 |
| persona_consistency | 0.4766 | 3 |
| naturalness | 0.5260 | 3 |
| overall_quality | 0.6002 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.8056 | 0.8667 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.6065 | 0.6667 |