# Publication Benchmark Artifact Report

- Run ID: `20260309T050127Z`
- Generated: `2026-03-09T05:02:41.807646+00:00`
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

- Reranker stage failed; see `retrieval/reranker_stage.json` for logs.

Retrieval by query type (lore / quest / persona / memory):
| Method | Query Type | Hit@k | MRR | nDCG@k |
|---|---|---:|---:|---:|
| bm25 | generic | 1.0000 | 0.8500 | 0.8893 |
| keyword_overlap | generic | 0.9000 | 0.8333 | 0.8500 |
| random | generic | 0.1000 | 0.1000 | 0.1000 |

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 669.829 (588.672, 799.727) | 936.747 (892.838, 963.155) | -266.918 |
| total_time_ms | 4544.448 (4420.795, 4694.938) | 4445.825 (3734.670, 4838.711) | 98.622 |
| tokens_per_s | 16.496 (16.426, 16.569) | 16.797 (16.611, 17.024) | -0.302 |
| bertscore_f1 | n/a | n/a | n/a |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| keyword_overlap | hit@5 | -0.1000 | -0.1000 |
| keyword_overlap | mrr | -0.0167 | -0.0196 |
| keyword_overlap | ndcg@5 | -0.0393 | -0.0442 |
| random | hit@5 | -0.9000 | -0.9000 |
| random | mrr | -0.7500 | -0.8824 |
| random | ndcg@5 | -0.7893 | -0.8875 |

## 4. Production Serving Baseline Comparison
The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) and identical generation settings (temperature and max tokens).
This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol.

## Claim Framing For Game Venues
Use quality + robustness under runtime constraints as the primary positioning claim.
Avoid framing this artifact as an overall best system across all speed/quality dimensions.
