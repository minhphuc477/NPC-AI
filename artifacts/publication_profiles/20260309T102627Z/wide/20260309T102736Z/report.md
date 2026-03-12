# Publication Benchmark Artifact Report

- Run ID: `20260309T102736Z`
- Generated: `2026-03-09T10:34:20.480924+00:00`
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

- Reranker stage failed; see `retrieval/reranker_stage.json` for logs.

Retrieval by query type (lore / quest / persona / memory):
| Method | Query Type | Hit@k | MRR | nDCG@k |
|---|---|---:|---:|---:|
| bm25 | generic | 1.0000 | 0.7333 | 0.8032 |
| keyword_overlap | generic | 1.0000 | 0.7306 | 0.8011 |
| random | generic | 0.0167 | 0.0043 | 0.0073 |

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 672.486 (647.907, 698.139) | 770.441 (751.299, 787.142) | -97.955 |
| total_time_ms | 4141.660 (4082.028, 4206.920) | 4136.654 (4079.818, 4194.836) | 5.006 |
| tokens_per_s | 18.617 (18.344, 18.886) | 19.226 (18.957, 19.482) | -0.609 |
| bertscore_f1 | n/a | n/a | n/a |

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

## Claim Framing For Game Venues
Use quality + robustness under runtime constraints as the primary positioning claim.
Avoid framing this artifact as an overall best system across all speed/quality dimensions.
