# Publication Benchmark Artifact Report

- Run ID: `20260225T061202Z`
- Generated: `2026-02-25T06:13:10.296758+00:00`
- Host: `http://127.0.0.1:11434`
- Candidate model: `elara-npc:latest`
- Baseline model: `phi3:mini`

## 1. Non-mock Benchmark Artifacts With Metadata
- Raw per-request serving traces are published in `serving/`.
- Hardware and model metadata are published in `metadata/`.
- All requests in this run were executed against live Ollama model endpoints (non-mock).

## 2. Standardized Retrieval Metrics (Labeled Sets)
| Method | Hit@k | MRR | nDCG@k |
|---|---:|---:|---:|
| bm25 | 1.0000 | 0.8500 | 0.8893 |
| keyword_overlap | 0.9000 | 0.8333 | 0.8500 |
| random | 0.4000 | 0.1650 | 0.2204 |
| reranker | 0.7000 | 0.3167 | 0.4123 |

- Reranker stage: 3360 hard-negative pairs, eval pair-accuracy=0.9563 (95% CI 0.9365, 0.9742).

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 842.265 (407.045, 1690.149) | 195.362 (151.912, 272.324) | 646.904 |
| total_time_ms | 2142.857 (1710.229, 2977.680) | 1505.360 (1407.370, 1628.542) | 637.497 |
| tokens_per_s | 24.826 (24.196, 25.579) | 24.694 (23.558, 25.781) | 0.132 |
| bertscore_f1 | 0.006 (-0.028, 0.035) | 0.139 (0.082, 0.190) | -0.134 |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| keyword_overlap | hit@5 | -0.1000 | -0.1000 |
| keyword_overlap | mrr | -0.0167 | -0.0196 |
| keyword_overlap | ndcg@5 | -0.0393 | -0.0442 |
| random | hit@5 | -0.6000 | -0.6000 |
| random | mrr | -0.6850 | -0.8059 |
| random | ndcg@5 | -0.6688 | -0.7521 |
| reranker | hit@5 | -0.3000 | -0.3000 |
| reranker | mrr | -0.5333 | -0.6275 |
| reranker | ndcg@5 | -0.4770 | -0.5363 |

## 4. Production Serving Baseline Comparison
The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) and identical generation settings (temperature and max tokens).
This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol.
