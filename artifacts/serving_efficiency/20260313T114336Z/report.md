# Serving Efficiency Matrix Report

- Run ID: `20260313T114336Z`
- Prompt count: `72`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 5383.428 | 13492.583 | 8.032 | nan | nan | no | nan |
| phi3:mini | 5164.447 | 12903.956 | 8.329 | nan | nan | no | nan |
| phi3:latest | 3471.474 | 11176.887 | 8.394 | nan | nan | no | nan |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | 588.627 | nan | nan |
| phi3:latest | 2315.696 | nan | nan |