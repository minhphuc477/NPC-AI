# Serving Efficiency Matrix Report

- Run ID: `20260225T002215Z`
- Prompt count: `48`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 656.318 | 4528.462 | 16.574 | -0.0258 | -0.0057 | no | 1.1458 |
| phi3:mini | 293.371 | 3952.260 | 17.454 | 0.0729 | 0.0184 | yes | 1.0000 |
| phi3:latest | 297.858 | 3964.982 | 17.579 | 0.0775 | 0.0195 | yes | 1.0000 |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | 576.202 | -0.0987 | -0.0241 |
| phi3:latest | 563.480 | -0.1032 | -0.0252 |