# Control Architecture Comparison

- Run root: `artifacts\proposal_control_tuning\architecture_compare_lock\20260306T164947Z`
- Profiles: `blend_balanced, runtime_optimized`
- Seeds: `29, 31`

| Profile | Quality Score | Operational Score | Delta OQ | Fallback Gap | Retry Gap | First-pass Gap |
|---|---:|---:|---:|---:|---:|---:|
| blend_balanced | -0.0054 | 0.0969 | -0.0039 | 0.0000 | -0.0156 | 0.1094 |
| runtime_optimized | -0.0157 | 0.0063 | -0.0109 | 0.0156 | -0.0156 | 0.0156 |