# Control Architecture Comparison

- Run root: `artifacts\proposal_control_tuning\architecture_compare_smoke3\20260306T084340Z`
- Profiles: `hybrid_balanced, intent_focus_adaptive, blend_balanced`
- Seeds: `31`

| Profile | Quality Score | Operational Score | Delta OQ | Fallback Gap | Retry Gap | First-pass Gap |
|---|---:|---:|---:|---:|---:|---:|
| blend_balanced | 0.0465 | 0.0000 | 0.0339 | 0.0000 | 0.0000 | 0.0000 |
| hybrid_balanced | -0.0279 | 0.6000 | -0.0195 | -0.2500 | -0.2500 | 0.2500 |
| intent_focus_adaptive | -0.1580 | -0.6000 | -0.1020 | 0.2500 | 0.2500 | -0.2500 |