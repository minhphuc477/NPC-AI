# Serving Efficiency Matrix Report

- Run ID: `20260313T143636Z`
- Prompt count: `72`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 1644.073 | 9583.113 | 8.264 | nan | nan | no | nan |
| phi3:mini | 428.746 | 8364.645 | 8.164 | nan | nan | no | nan |
| phi3:latest | 447.055 | 8363.240 | 8.160 | nan | nan | no | nan |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | 1218.468 | nan | nan |
| phi3:latest | 1219.873 | nan | nan |