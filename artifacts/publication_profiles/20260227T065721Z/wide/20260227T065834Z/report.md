# Publication Benchmark Artifact Report

- Run ID: `20260227T065834Z`
- Generated: `2026-02-27T07:01:57.579124+00:00`
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
| ttft_ms | 271.631 (237.183, 325.128) | 140.110 (128.390, 154.407) | 131.521 |
| total_time_ms | 1714.869 (1676.992, 1776.556) | 1542.421 (1521.637, 1562.456) | 172.448 |
| tokens_per_s | 39.200 (38.877, 39.478) | 40.101 (39.949, 40.250) | -0.901 |
| bertscore_f1 | -0.029 (-0.043, -0.015) | 0.071 (0.061, 0.081) | -0.100 |

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
