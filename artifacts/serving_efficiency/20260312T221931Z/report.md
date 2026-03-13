# Serving Efficiency Matrix Report

- Run ID: `20260312T221931Z`
- Prompt count: `72`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 673.555 | 4487.470 | 16.924 | nan | nan | no | nan |
| phi3:mini | 222.637 | 3073.148 | 22.871 | nan | nan | no | nan |
| phi3:latest | 216.905 | 2956.541 | 23.613 | nan | nan | no | nan |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | 1414.321 | nan | nan |
| phi3:latest | 1530.929 | nan | nan |