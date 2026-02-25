# Publication Benchmark Artifact Report

- Run ID: `20260225T071334Z`
- Generated: `2026-02-25T07:14:11.842460+00:00`
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
| ttft_ms | 806.845 (345.819, 1670.739) | 181.340 (149.894, 234.411) | 625.505 |
| total_time_ms | 1072.687 (604.017, 1942.557) | 465.248 (439.347, 509.617) | 607.439 |
| tokens_per_s | 30.498 (28.871, 32.084) | 28.472 (27.402, 29.654) | 2.026 |
| bertscore_f1 | -0.076 (-0.108, -0.043) | -0.033 (-0.085, 0.019) | -0.043 |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| n/a | n/a | n/a | n/a |

## 4. Production Serving Baseline Comparison
The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) and identical generation settings (temperature and max tokens).
This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol.
