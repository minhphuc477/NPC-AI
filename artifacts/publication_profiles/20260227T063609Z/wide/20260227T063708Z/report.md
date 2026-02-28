# Publication Benchmark Artifact Report

- Run ID: `20260227T063708Z`
- Generated: `2026-02-27T06:40:29.285822+00:00`
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
| bm25 | 1.0000 | 0.7333 | 0.8032 |
| keyword_overlap | 1.0000 | 0.7306 | 0.8011 |
| random | 0.0167 | 0.0043 | 0.0073 |
| reranker | 1.0000 | 0.8000 | 0.8524 |

- Reranker stage: 3360 hard-negative pairs, eval pair-accuracy=0.9286 (95% CI 0.9048, 0.9504).

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 283.248 (253.619, 326.705) | 151.143 (140.458, 163.525) | 132.105 |
| total_time_ms | 1711.288 (1675.379, 1762.068) | 1639.264 (1607.888, 1672.897) | 72.024 |
| tokens_per_s | 39.656 (39.217, 40.039) | 38.208 (37.405, 38.937) | 1.447 |
| bertscore_f1 | -0.030 (-0.042, -0.018) | 0.075 (0.063, 0.087) | -0.105 |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| keyword_overlap | hit@5 | 0.0000 | 0.0000 |
| keyword_overlap | mrr | -0.0028 | -0.0038 |
| keyword_overlap | ndcg@5 | -0.0021 | -0.0026 |
| random | hit@5 | -0.9833 | -0.9833 |
| random | mrr | -0.7290 | -0.9941 |
| random | ndcg@5 | -0.7959 | -0.9909 |
| reranker | hit@5 | 0.0000 | 0.0000 |
| reranker | mrr | 0.0667 | 0.0909 |
| reranker | ndcg@5 | 0.0492 | 0.0613 |

## 4. Production Serving Baseline Comparison
The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) and identical generation settings (temperature and max tokens).
This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol.
