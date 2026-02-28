# Serving Efficiency Matrix Report

- Run ID: `20260227T063202Z`
- Prompt count: `48`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 260.139 | 1637.429 | 40.690 | -0.0347 | -0.0212 | no | 1.0819 |
| phi3:mini | 156.950 | 1522.900 | 41.351 | 0.0825 | 0.0542 | yes | 1.0000 |
| phi3:latest | 149.891 | 1513.485 | 41.248 | 0.0736 | 0.0486 | yes | 1.0000 |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | 114.528 | -0.1172 | -0.0754 |
| phi3:latest | 123.944 | -0.1083 | -0.0698 |