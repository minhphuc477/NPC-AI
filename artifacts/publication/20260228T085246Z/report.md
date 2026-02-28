# Publication Benchmark Artifact Report

- Run ID: `20260228T085246Z`
- Generated: `2026-02-28T08:53:47.882704+00:00`
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
| reranker | 0.9000 | 0.8000 | 0.8262 |

- Reranker stage: 3360 hard-negative pairs, eval pair-accuracy=0.9563 (95% CI 0.9365, 0.9742).

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 284.849 (262.316, 314.787) | 153.444 (131.788, 187.921) | 131.405 |
| total_time_ms | 1837.895 (1609.015, 1979.257) | 1753.152 (1707.029, 1797.690) | 84.744 |
| tokens_per_s | 39.284 (38.697, 39.889) | 40.266 (39.949, 40.548) | -0.982 |
| bertscore_f1 | -0.011 (-0.072, 0.039) | 0.142 (0.083, 0.208) | -0.153 |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| keyword_overlap | hit@5 | -0.1000 | -0.1000 |
| keyword_overlap | mrr | -0.0167 | -0.0196 |
| keyword_overlap | ndcg@5 | -0.0393 | -0.0442 |
| random | hit@5 | -0.6000 | -0.6000 |
| random | mrr | -0.6850 | -0.8059 |
| random | ndcg@5 | -0.6688 | -0.7521 |
| reranker | hit@5 | -0.1000 | -0.1000 |
| reranker | mrr | -0.0500 | -0.0588 |
| reranker | ndcg@5 | -0.0631 | -0.0709 |

## 4. Production Serving Baseline Comparison
The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) and identical generation settings (temperature and max tokens).
This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol.
