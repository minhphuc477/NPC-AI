# Human Evaluation Report

- Total normalized rows: `324`
- Metrics: `context_relevance, persona_consistency, naturalness, overall_quality`

## Arm Summary
| Arm | Metric | Mean | 95% CI |
|---|---|---:|---:|
| baseline_no_context | context_relevance | 0.6444 | (0.6000, 0.6907) |
| baseline_no_context | persona_consistency | 0.7889 | (0.7537, 0.8204) |
| baseline_no_context | naturalness | 0.7352 | (0.7037, 0.7648) |
| baseline_no_context | overall_quality | 0.6926 | (0.6574, 0.7278) |
| baseline_no_context_phi3_latest | context_relevance | 0.7093 | (0.6741, 0.7482) |
| baseline_no_context_phi3_latest | persona_consistency | 0.7704 | (0.7333, 0.8093) |
| baseline_no_context_phi3_latest | naturalness | 0.7907 | (0.7556, 0.8259) |
| baseline_no_context_phi3_latest | overall_quality | 0.7333 | (0.7019, 0.7667) |
| proposed_contextual_controlled | context_relevance | 0.7944 | (0.7722, 0.8149) |
| proposed_contextual_controlled | persona_consistency | 0.8833 | (0.8500, 0.9130) |
| proposed_contextual_controlled | naturalness | 0.6556 | (0.6259, 0.6870) |
| proposed_contextual_controlled | overall_quality | 0.7704 | (0.7444, 0.7944) |

## Agreement
| Metric | Mean Pairwise Kappa | Pair Count |
|---|---:|---:|
| context_relevance | 0.4951 | 3 |
| persona_consistency | 0.4186 | 3 |
| naturalness | 0.3890 | 3 |
| overall_quality | 0.3786 | 3 |

## Preference Wins
| Comparison | Soft Win Rate | Strict Non-tie Win Rate |
|---|---:|---:|
| proposed_contextual_controlled_vs_baseline_no_context | 0.6250 | 0.6753 |
| proposed_contextual_controlled_vs_baseline_no_context_phi3_latest | 0.5556 | 0.5968 |