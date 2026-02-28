# Serving Efficiency Matrix Report

- Run ID: `20260228T085349Z`
- Prompt count: `48`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 256.478 | 1904.983 | 39.430 | -0.0457 | -0.0240 | no | 1.0717 |
| phi3:mini | 150.412 | 1777.612 | 39.966 | 0.0728 | 0.0409 | yes | 1.0000 |
| phi3:latest | 152.167 | 1785.211 | 39.630 | 0.0705 | 0.0395 | no | 1.0043 |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | 127.371 | -0.1185 | -0.0649 |
| phi3:latest | 119.772 | -0.1163 | -0.0635 |