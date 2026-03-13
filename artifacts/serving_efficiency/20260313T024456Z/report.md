# Serving Efficiency Matrix Report

- Run ID: `20260313T024456Z`
- Prompt count: `72`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 518.411 | 3156.934 | 21.510 | nan | nan | no | nan |
| phi3:mini | 205.396 | 2506.541 | 24.672 | nan | nan | no | nan |
| phi3:latest | 209.179 | 2522.206 | 24.548 | nan | nan | no | nan |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | 650.393 | nan | nan |
| phi3:latest | 634.729 | nan | nan |