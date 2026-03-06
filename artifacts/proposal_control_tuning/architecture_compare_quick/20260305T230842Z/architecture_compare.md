# Control Architecture Comparison

- Run root: `artifacts\proposal_control_tuning\architecture_compare_quick\20260305T230842Z`
- Profiles: `runtime_optimized, hybrid_balanced, intent_focus_adaptive`
- Seeds: `29`

| Profile | Quality Score | Operational Score | Delta OQ | Fallback Gap | Retry Gap | First-pass Gap |
|---|---:|---:|---:|---:|---:|---:|
| intent_focus_adaptive | -0.0095 | 0.0000 | -0.0084 | 0.0000 | 0.0000 | 0.0000 |
| hybrid_balanced | -0.0133 | 0.2500 | -0.0083 | -0.2500 | 0.0000 | 0.0000 |
| runtime_optimized | -0.0211 | -0.2000 | -0.0134 | 0.0833 | 0.0833 | -0.0833 |