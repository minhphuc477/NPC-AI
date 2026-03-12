# Serving Efficiency Matrix Report

- Run ID: `20260312T112954Z`
- Prompt count: `72`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 19560.481 | 26926.437 | 9.173 | nan | nan | no | nan |
| phi3:mini | 26599.798 | 33162.854 | 10.168 | nan | nan | no | nan |
| phi3:latest | 34421.159 | 40934.717 | 9.953 | nan | nan | no | nan |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | -6236.417 | nan | nan |
| phi3:latest | -14008.281 | nan | nan |