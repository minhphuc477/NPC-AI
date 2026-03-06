# Verification Checklist (Current)

## Verified On Latest Runs
- Proposal: `artifacts/proposal/20260302T182844Z`
- Publication: `artifacts/publication/20260302T191131Z`
- Gate report: `artifacts/proposal/20260302T182844Z/quality_gate_report.json`

## Pass/Fail Summary
| Check | Status |
|---|---|
| Proposal scenario coverage >= 100 | Pass |
| Controlled significant gains on key metrics | Pass |
| External significant wins threshold | Pass |
| Human-eval row count/agreement/preferences | Pass |
| Publication metadata completeness | Pass |
| Retrieval standardized metrics + ablations | Pass |
| Serving CIs + prompt parity | Pass |
| Security benchmark thresholds | Pass |

## Notes
- Overall strict gate status: `overall_pass=true`.
- Serving-efficiency superiority is not part of gate pass criteria and remains an open improvement item.

## Re-run Verification
```bash
python scripts/proposal_quality_gate.py --proposal-run 20260302T182844Z --publication-run 20260302T191131Z --require-human-eval --require-security-benchmark --strict
```
