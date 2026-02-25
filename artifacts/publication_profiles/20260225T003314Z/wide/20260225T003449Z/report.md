# Publication Benchmark Artifact Report

- Run ID: `20260225T003449Z`
- Generated: `2026-02-25T00:42:14.203490+00:00`
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

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 673.247 (619.562, 720.521) | 298.825 (266.380, 339.149) | 374.422 |
| total_time_ms | 4652.619 (4505.435, 4809.488) | 3986.401 (3907.269, 4081.323) | 666.218 |
| tokens_per_s | 16.285 (15.853, 16.683) | 17.517 (17.185, 17.800) | -1.232 |
| bertscore_f1 | -0.028 (-0.038, -0.017) | 0.076 (0.066, 0.084) | -0.103 |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| keyword_overlap | hit@5 | 0.0000 | 0.0000 |
| keyword_overlap | mrr | -0.0028 | -0.0038 |
| keyword_overlap | ndcg@5 | -0.0021 | -0.0026 |
| random | hit@5 | -0.9833 | -0.9833 |
| random | mrr | -0.7290 | -0.9941 |
| random | ndcg@5 | -0.7959 | -0.9909 |

## 4. Production Serving Baseline Comparison
The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) and identical generation settings (temperature and max tokens).
This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol.
