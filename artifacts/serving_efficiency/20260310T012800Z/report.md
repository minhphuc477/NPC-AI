# Serving Efficiency Matrix Report

- Run ID: `20260310T012800Z`
- Prompt count: `72`
- Models: `elara-npc:latest, phi3:mini, phi3:latest`

## Summary
| Model | TTFT ms | Total ms | Tokens/s | BERTScore F1 | QPS-like (quality/s) | Pareto | Frontier ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| elara-npc:latest | 725.453 | 4322.397 | 17.870 | nan | nan | no | nan |
| phi3:mini | 232.535 | 3922.161 | 17.643 | nan | nan | no | nan |
| phi3:latest | 249.420 | 4004.541 | 17.206 | nan | nan | no | nan |

## Candidate Delta vs Baselines
| Baseline | Delta total ms | Delta BERTScore | Delta quality/s |
|---|---:|---:|---:|
| phi3:mini | 400.235 | nan | nan |
| phi3:latest | 317.855 | nan | nan |