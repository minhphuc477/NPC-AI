# Ablation Report

## 1. Purpose
Measure delta from fixed baselines with reproducible artifact outputs.

## 2. Current Published Ablations
Run ID: `20260224T101457Z`

Files:
- `artifacts/publication/20260224T101457Z/retrieval/metrics.json`
- `artifacts/publication/20260224T101457Z/retrieval/ablation_deltas_vs_bm25.json`
- `artifacts/publication/20260224T101457Z/serving/delta_vs_baseline.json`
- `artifacts/publication/20260224T101457Z/retrieval/security_guard_benchmark.json`

Included ablation comparisons:
- Retrieval baseline: `bm25`
- Retrieval ablations: `keyword_overlap`, `random`
- Serving baseline: `phi3:mini`
- Serving candidate: `elara-npc:latest`
- Security baseline: robustness guard OFF
- Security ablation: robustness guard ON

## 3. Confidence Intervals
95% bootstrap confidence intervals are published per metric in:
- `retrieval/metrics.json`
- `serving/summary.json`

## 4. Interpretation
- Retrieval ablations behave as expected (`bm25` > `keyword_overlap` > `random`).
- Serving comparison does not currently show candidate latency/throughput gains
  over baseline in this run.
- Retrieval security benchmark shows full ASR suppression in the synthetic
  poisoned benchmark set (1.0 -> 0.0).

