# Publication Benchmark Artifact Report

- Run ID: `20260310T011758Z`
- Generated: `2026-03-10T01:27:59.617630+00:00`
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
| bm25 | 0.9931 | 0.6972 | 0.6390 |
| keyword_overlap | 0.9965 | 0.6997 | 0.6413 |
| random | 0.0174 | 0.0095 | 0.0058 |

- Reranker stage failed; see `retrieval/reranker_stage.json` for logs.

Retrieval by query type (lore / quest / persona / memory):
| Method | Query Type | Hit@k | MRR | nDCG@k |
|---|---|---:|---:|---:|
| bm25 | lore | 0.9722 | 0.6590 | 0.6107 |
| bm25 | memory | 1.0000 | 0.7130 | 0.6503 |
| bm25 | persona | 1.0000 | 0.7060 | 0.6460 |
| bm25 | quest | 1.0000 | 0.7106 | 0.6490 |
| keyword_overlap | lore | 0.9861 | 0.6817 | 0.6280 |
| keyword_overlap | memory | 1.0000 | 0.7130 | 0.6503 |
| keyword_overlap | persona | 1.0000 | 0.7060 | 0.6460 |
| keyword_overlap | quest | 1.0000 | 0.6979 | 0.6410 |
| random | lore | 0.0278 | 0.0167 | 0.0130 |
| random | memory | 0.0139 | 0.0028 | 0.0044 |
| random | persona | 0.0139 | 0.0139 | 0.0038 |
| random | quest | 0.0139 | 0.0046 | 0.0019 |

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 751.737 (695.293, 805.923) | 227.733 (190.182, 272.049) | 524.004 |
| total_time_ms | 4454.169 (4366.488, 4561.542) | 3764.554 (3720.772, 3812.820) | 689.615 |
| tokens_per_s | 17.510 (17.253, 17.721) | 18.279 (18.183, 18.383) | -0.769 |
| bertscore_f1 | n/a | n/a | n/a |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| keyword_overlap | hit@5 | 0.0035 | 0.0035 |
| keyword_overlap | mrr | 0.0025 | 0.0036 |
| keyword_overlap | ndcg@5 | 0.0023 | 0.0037 |
| random | hit@5 | -0.9757 | -0.9825 |
| random | mrr | -0.6877 | -0.9864 |
| random | ndcg@5 | -0.6332 | -0.9909 |

## 4. Production Serving Baseline Comparison
The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) and identical generation settings (temperature and max tokens).
This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol.

## Claim Framing For Game Venues
Use quality + robustness under runtime constraints as the primary positioning claim.
Avoid framing this artifact as an overall best system across all speed/quality dimensions.
