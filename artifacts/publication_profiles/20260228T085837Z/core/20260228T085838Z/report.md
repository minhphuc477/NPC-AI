# Publication Benchmark Artifact Report

- Run ID: `20260228T085838Z`
- Generated: `2026-02-28T08:59:36.953364+00:00`
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
| random | 0.1000 | 0.1000 | 0.1000 |
| reranker | 0.9000 | 0.8000 | 0.8262 |

- Reranker stage: 3360 hard-negative pairs, eval pair-accuracy=0.8948 (95% CI 0.8690, 0.9206).

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 314.246 (247.816, 434.985) | 147.282 (125.508, 181.980) | 166.964 |
| total_time_ms | 1945.729 (1878.504, 2068.954) | 1734.215 (1650.771, 1796.976) | 211.515 |
| tokens_per_s | 39.537 (39.372, 39.681) | 39.659 (39.322, 39.971) | -0.121 |
| bertscore_f1 | 0.002 (-0.042, 0.042) | 0.130 (0.061, 0.194) | -0.128 |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| keyword_overlap | hit@5 | -0.1000 | -0.1000 |
| keyword_overlap | mrr | -0.0167 | -0.0196 |
| keyword_overlap | ndcg@5 | -0.0393 | -0.0442 |
| random | hit@5 | -0.9000 | -0.9000 |
| random | mrr | -0.7500 | -0.8824 |
| random | ndcg@5 | -0.7893 | -0.8875 |
| reranker | hit@5 | -0.1000 | -0.1000 |
| reranker | mrr | -0.0500 | -0.0588 |
| reranker | ndcg@5 | -0.0631 | -0.0709 |

## 4. Production Serving Baseline Comparison
The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) and identical generation settings (temperature and max tokens).
This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol.
