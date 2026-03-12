# Serving Efficiency Matrix Report

- Run ID: `20260309T044432Z`
- Prompt count: `72`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 669.086 | 4539.625 | 16.495 | nan | nan | no | nan |
| phi3:mini | 938.332 | 4757.112 | 16.835 | nan | nan | no | nan |
| phi3:latest | 936.978 | 4774.471 | 16.744 | nan | nan | no | nan |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | -217.487 | nan | nan |
| phi3:latest | -234.846 | nan | nan |