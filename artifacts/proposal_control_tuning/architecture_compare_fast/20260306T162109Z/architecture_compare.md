# Control Architecture Comparison

- Run root: `artifacts\proposal_control_tuning\architecture_compare_fast\20260306T162109Z`
- Profiles: `runtime_optimized, hybrid_balanced, intent_focus_adaptive, blend_balanced`
- Seeds: `29`

| Profile | Quality Score | Operational Score | Delta OQ | Fallback Gap | Retry Gap | First-pass Gap |
|---|---:|---:|---:|---:|---:|---:|
| blend_balanced | 0.0112 | 0.1200 | 0.0072 | 0.0000 | 0.0000 | 0.1500 |
| runtime_optimized | -0.0012 | -0.0700 | -0.0003 | 0.0000 | 0.0500 | -0.0500 |
| hybrid_balanced | -0.0037 | 0.0000 | -0.0021 | 0.0000 | 0.0000 | 0.0000 |
| intent_focus_adaptive | -0.0055 | 0.0000 | -0.0044 | 0.0000 | 0.0000 | 0.0000 |