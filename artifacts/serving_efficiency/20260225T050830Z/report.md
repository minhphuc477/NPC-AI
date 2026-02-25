# Serving Efficiency Matrix Report

- Run ID: `20260225T050830Z`
- Prompt count: `48`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 696.984 | 3930.904 | 20.097 | -0.0385 | -0.0098 | no | 1.3637 |
| phi3:mini | 252.456 | 4331.811 | 20.285 | 0.0731 | 0.0169 | yes | 1.0000 |
| phi3:latest | 197.002 | 2882.530 | 23.972 | 0.0692 | 0.0240 | yes | 1.0000 |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | -400.907 | -0.1116 | -0.0267 |
| phi3:latest | 1048.374 | -0.1077 | -0.0338 |