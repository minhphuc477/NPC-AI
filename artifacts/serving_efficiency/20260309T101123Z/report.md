# Serving Efficiency Matrix Report

- Run ID: `20260309T101123Z`
- Prompt count: `72`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 743.880 | 4376.384 | 17.903 | nan | nan | no | nan |
| phi3:mini | 753.284 | 4093.409 | 19.337 | nan | nan | no | nan |
| phi3:latest | 752.781 | 4066.280 | 19.547 | nan | nan | no | nan |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | 282.975 | nan | nan |
| phi3:latest | 310.104 | nan | nan |