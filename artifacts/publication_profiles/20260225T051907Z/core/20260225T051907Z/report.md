# Publication Benchmark Artifact Report

- Run ID: `20260225T051907Z`
- Generated: `2026-02-25T05:20:20.995432+00:00`
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

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 445.255 (395.819, 524.882) | 201.446 (158.434, 278.468) | 243.809 |
| total_time_ms | 3264.315 (3167.269, 3371.964) | 2951.695 (2829.349, 3063.522) | 312.620 |
| tokens_per_s | 22.890 (22.366, 23.520) | 23.000 (22.372, 23.788) | -0.110 |
| bertscore_f1 | -0.001 (-0.040, 0.038) | 0.144 (0.072, 0.218) | -0.144 |

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
