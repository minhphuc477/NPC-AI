# Final Benchmark Report (Current Unified Artifact)

## Canonical Runs
- Proposal run: `artifacts/proposal/20260302T182844Z`
- Publication run: `artifacts/publication/20260302T191131Z`
- Strict gate: `artifacts/proposal/20260302T182844Z/quality_gate_report.json`

## Executive Summary
The strict quality gate passes with `overall_pass=true`. The project has strong, reproducible gains on controlled context-grounded quality and retrieval security robustness, with a known serving-speed tradeoff.

## Proposal-Critical Results
- Scenario coverage: 112 (required >= 100).
- Controlled vs raw improvements (all significant):
  - context relevance +0.2131
  - persona consistency +0.2150
  - naturalness +0.1158
  - overall quality +0.1808
- External baseline threshold: 10/12 significant-positive metrics vs each baseline.

## Human Evaluation
- 324 ratings, 3 raters.
- Mean pairwise kappa: 0.5329.
- Preference soft win rate:
  - 0.7315 vs `phi3:mini`
  - 0.6806 vs `phi3:latest`

## Publication Benchmark Checks
1. Non-mock artifact metadata: present.
2. Standard retrieval metrics (Hit@k/MRR/nDCG): present.
3. Confidence intervals and ablation deltas: present.
4. Prompt parity for serving comparison: verified.
5. Security benchmark thresholds: passed.

## Security Robustness
Poisoned + trust-spoofed stress tests (100 scenarios each):
- baseline ASR 1.0000
- guarded ASR 0.0000
- relative reduction 1.0000

## Known Weakness
Serving efficiency superiority is not shown against `phi3:mini` under identical settings.

## Reproduce
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```
