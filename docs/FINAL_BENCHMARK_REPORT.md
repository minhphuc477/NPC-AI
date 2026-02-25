# Final Benchmark Report

## Scope
Consolidated snapshot of the latest full checkout artifacts and supported claims.

## Latest Unified Checkout
- Manifest: `artifacts/final_checkout/20260225T052614Z/manifest.json`
- Proposal run: `artifacts/proposal/20260224T175344Z`
- Publication run: `artifacts/publication/20260224T151628Z`
- Serving matrix run: `artifacts/serving_efficiency/20260225T050830Z`
- External profile suite: `artifacts/publication_profiles/20260225T051907Z`

## Quality Gate
- Final gate report: `artifacts/proposal/20260224T175344Z/quality_gate_report_final.md`
- Status: `PASS` (with `--require-human-eval`)

## Core Evidence
1. Non-mock publication artifacts with metadata:
- `metadata/hardware.json`
- `metadata/models.json`

2. Retrieval metrics (standardized):
- `retrieval/metrics.json`
- `retrieval/ablation_deltas_vs_bm25.json`

3. Human evaluation (completed multi-rater attachment):
- `human_eval_llm_multirater_consistent.csv`
- `human_eval_summary.json`
- `human_eval_report.md`

4. Lexical diversity:
- `lexical_diversity_summary.json`
- `lexical_diversity_report.md`

5. Serving quality/efficiency frontier:
- `artifacts/serving_efficiency/20260225T050830Z/summary.json`
- `artifacts/serving_efficiency/20260225T050830Z/report.md`

6. Wider retrieval coverage + hard negatives:
- `data/retrieval_gold_wide.jsonl`
- `data/retrieval_hard_negatives_wide.jsonl`
- `data/retrieval_reranker_pairs_wide.jsonl`

## Supported Claims
1. Strongly supported:
- context-grounded dialogue quality gains under response control
- retrieval robustness advantage under poisoning-style stress
- completed publication-grade pipeline artifacts (with CIs/deltas and human-eval attachment)

2. Not supported as superiority claim:
- serving latency/throughput dominance on quality-normalized frontier

## Reproduce
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```
