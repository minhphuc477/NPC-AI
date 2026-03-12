# Proposal Multi-Seed Aggregate

- Generated: `2026-03-10T12:11:36.155552+00:00`
- Proposal root: `artifacts\proposal`
- Target arm: `proposed_contextual_controlled`
- Runs included: `2`

## Included Runs
- `20260310T003216Z` -> `artifacts\proposal\20260310T003216Z`
- `20260309T115334Z` -> `artifacts\proposal\20260309T115334Z`

## Arm Metric Aggregates
| Metric | Mean | Std | 95% CI | Min | Max | N |
|---|---:|---:|---:|---:|---:|---:|
| overall_quality | 0.392193 | 0.022160 | (0.361481, 0.422905) | 0.376524 | 0.407863 | 2 |
| context_relevance | 0.238786 | 0.045048 | (0.176353, 0.301219) | 0.206933 | 0.270640 | 2 |
| persona_consistency | 0.298321 | 0.001598 | (0.296106, 0.300535) | 0.297191 | 0.299451 | 2 |
| naturalness | 0.898885 | 0.001972 | (0.896152, 0.901618) | 0.897490 | 0.900279 | 2 |
| quest_state_correctness | 0.295198 | 0.040826 | (0.238615, 0.351780) | 0.266329 | 0.324066 | 2 |
| lore_consistency | 0.157399 | 0.045611 | (0.094186, 0.220612) | 0.125147 | 0.189651 | 2 |
| multi_turn_contradiction_safety | 1.000000 | 0.000000 | (1.000000, 1.000000) | 1.000000 | 1.000000 | 2 |
| objective_completion_support | 0.072818 | 0.018995 | (0.046492, 0.099144) | 0.059387 | 0.086250 | 2 |
| gameplay_usefulness | 0.355042 | 0.001065 | (0.353567, 0.356517) | 0.354289 | 0.355795 | 2 |
| time_pressure_acceptability | 0.291519 | 0.004097 | (0.285842, 0.297197) | 0.288623 | 0.294416 | 2 |

## Operational Aggregates
| Metric | Mean | Std | 95% CI | Min | Max | N |
|---|---:|---:|---:|---:|---:|---:|
| fallback_rate | 0.006944 | 0.009821 | (-0.006667, 0.020556) | 0.000000 | 0.013889 | 2 |
| retry_rate | 0.083333 | 0.117851 | (-0.080000, 0.246667) | 0.000000 | 0.166667 | 2 |
| first_pass_accept_rate | 0.593750 | 0.220971 | (0.287500, 0.900000) | 0.437500 | 0.750000 | 2 |
| timeout_rate | 0.000000 | 0.000000 | (0.000000, 0.000000) | 0.000000 | 0.000000 | 2 |
| error_rate | 0.000000 | 0.000000 | (0.000000, 0.000000) | 0.000000 | 0.000000 | 2 |
