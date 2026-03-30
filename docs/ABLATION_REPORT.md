# Ablation Report (Current)

## Run References
- Proposal run: `storage/artifacts/proposal/20260302T182844Z`
- Publication run: `storage/artifacts/publication/20260302T191131Z`

## A. Response-Control Ablation (Core Proposal)
Comparison: `controlled_vs_proposed_raw` (same base model, dynamic context on; only response control toggled).

| Metric | Mean Delta | 95% CI | p(delta<=0) |
|---|---:|---:|---:|
| context_relevance | +0.2131 | (0.1962, 0.2285) | 0.0000 |
| persona_consistency | +0.2150 | (0.1878, 0.2419) | 0.0000 |
| naturalness | +0.1158 | (0.0964, 0.1337) | 0.0000 |
| overall_quality | +0.1808 | (0.1666, 0.1953) | 0.0000 |
| bertscore_f1 | +0.0277 | (0.0131, 0.0422) | 0.0000 |
| distinct1 | +0.0072 | (-0.0029, 0.0185) | 0.0853 |

Interpretation: response control is the strongest contributor to proposal-critical gains.

## B. Retrieval Method Ablation (Publication)
Baseline retrieval method: BM25.

| Method | Hit@5 Delta | MRR Delta | nDCG@5 Delta |
|---|---:|---:|---:|
| keyword_overlap | -0.1000 | -0.0167 | -0.0393 |
| random | -0.6000 | -0.6850 | -0.6688 |
| reranker | -0.1000 | -0.0500 | -0.0631 |

Interpretation: on the current 10-query publication core set, BM25 is strongest on standard IR metrics.

## C. Serving Ablation vs Baseline Runtime
Candidate: `elara-npc:latest` vs baseline `phi3:mini`.

| Metric | Candidate | Baseline | Delta | Better? |
|---|---:|---:|---:|---|
| ttft_ms | 542.62 | 273.86 | +268.76 | No |
| total_time_ms | 3601.54 | 3138.41 | +463.13 | No |
| tokens_per_s | 18.45 | 18.86 | -0.41 | No |

Interpretation: no serving-efficiency win in current artifact.

## How to Run With Ablation Option
- Full run with ablations included:
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```
- Faster run skipping ablation baselines:
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434 --skip-ablation-baselines
```

