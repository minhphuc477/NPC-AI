# Publication Benchmark Artifact Report

- Run ID: `20260225T071636Z`
- Generated: `2026-02-25T07:17:07.845502+00:00`
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

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 452.022 (354.235, 604.226) | 184.012 (152.503, 240.705) | 268.009 |
| total_time_ms | 579.639 (483.406, 731.987) | 311.049 (274.789, 370.994) | 268.591 |
| tokens_per_s | 31.963 (30.513, 33.757) | 31.652 (29.884, 33.225) | 0.311 |
| bertscore_f1 | -0.147 (-0.191, -0.104) | -0.074 (-0.129, -0.016) | -0.074 |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| n/a | n/a | n/a | n/a |

## 4. Production Serving Baseline Comparison
The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) and identical generation settings (temperature and max tokens).
This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol.
