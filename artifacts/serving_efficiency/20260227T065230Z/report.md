# Serving Efficiency Matrix Report

- Run ID: `20260227T065230Z`
- Prompt count: `48`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 284.329 | 1915.875 | 34.649 | -0.0302 | -0.0158 | no | 1.0882 |
| phi3:mini | 157.584 | 1760.516 | 35.065 | 0.0810 | 0.0460 | yes | 1.0000 |
| phi3:latest | 152.104 | 1800.276 | 34.250 | 0.0730 | 0.0406 | no | 1.0226 |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | 155.359 | -0.1112 | -0.0618 |
| phi3:latest | 115.599 | -0.1033 | -0.0564 |