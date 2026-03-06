# Control Architecture Comparison

- Run root: `artifacts\proposal_control_tuning\architecture_compare_seeded\20260305T233017Z`
- Profiles: `hybrid_balanced, intent_focus_adaptive`
- Seeds: `29, 31`

| Profile | Quality Score | Operational Score | Delta OQ | Fallback Gap | Retry Gap | First-pass Gap |
|---|---:|---:|---:|---:|---:|---:|
| hybrid_balanced | 0.0194 | 0.1125 | 0.0132 | -0.0833 | -0.0208 | 0.0208 |
| intent_focus_adaptive | 0.0116 | -0.0583 | 0.0075 | 0.0000 | 0.0417 | -0.0417 |