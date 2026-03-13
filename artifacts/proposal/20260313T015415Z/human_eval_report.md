# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.5815 | (0.5481, 0.6130) |
| baseline_no_context | persona_consistency | 0.7704 | (0.7407, 0.8000) |
| baseline_no_context | naturalness | 0.7130 | (0.6889, 0.7389) |
| baseline_no_context | overall_quality | 0.6963 | (0.6685, 0.7241) |
| baseline_no_context_phi3_latest | context_relevance | 0.6444 | (0.6093, 0.6833) |
| baseline_no_context_phi3_latest | persona_consistency | 0.8389 | (0.8019, 0.8759) |
| baseline_no_context_phi3_latest | naturalness | 0.7852 | (0.7500, 0.8185) |
| baseline_no_context_phi3_latest | overall_quality | 0.7093 | (0.6815, 0.7370) |
| proposed_contextual_controlled_tuned | context_relevance | 0.8056 | (0.7833, 0.8278) |
| proposed_contextual_controlled_tuned | persona_consistency | 0.8778 | (0.8481, 0.9056) |
| proposed_contextual_controlled_tuned | naturalness | 0.7593 | (0.7241, 0.7907) |
| proposed_contextual_controlled_tuned | overall_quality | 0.8241 | (0.7907, 0.8537) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.7267 | 3 |
| persona_consistency | 0.6869 | 3 |
| naturalness | 0.6234 | 3 |
| overall_quality | 0.6546 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_tuned_vs_baseline_no_context | 0.8009 | 0.8736 |
| proposed_contextual_controlled_tuned_vs_baseline_no_context_phi3_latest | 0.6713 | 0.7284 |