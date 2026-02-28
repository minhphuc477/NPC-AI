# Publication Benchmark Artifact Report

- Run ID: `20260227T065119Z`
- Generated: `2026-02-27T06:52:28.638275+00:00`
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
| random | 0.4000 | 0.1650 | 0.2204 |
| reranker | 0.9000 | 0.8000 | 0.8262 |

- Reranker stage: 3360 hard-negative pairs, eval pair-accuracy=0.9563 (95% CI 0.9365, 0.9742).

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 672.028 (307.426, 1379.395) | 165.359 (138.952, 210.052) | 506.668 |
| total_time_ms | 2294.889 (1912.336, 3027.688) | 1634.796 (1335.812, 1813.402) | 660.093 |
| tokens_per_s | 34.882 (34.232, 35.492) | 35.396 (34.959, 35.949) | -0.514 |
| bertscore_f1 | 0.020 (-0.013, 0.053) | 0.135 (0.080, 0.193) | -0.115 |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| keyword_overlap | hit@5 | -0.1000 | -0.1000 |
| keyword_overlap | mrr | -0.0167 | -0.0196 |
| keyword_overlap | ndcg@5 | -0.0393 | -0.0442 |
| random | hit@5 | -0.6000 | -0.6000 |
| random | mrr | -0.6850 | -0.8059 |
| random | ndcg@5 | -0.6688 | -0.7521 |
| reranker | hit@5 | -0.1000 | -0.1000 |
| reranker | mrr | -0.0500 | -0.0588 |
| reranker | ndcg@5 | -0.0631 | -0.0709 |

## 4. Production Serving Baseline Comparison
The candidate and baseline were benchmarked under identical prompts (`data/serving_prompts.jsonl`) and identical generation settings (temperature and max tokens).
This satisfies a production-serving baseline comparison requirement with a fixed dataset and fixed prompt protocol.

## 5. Adversarial Retrieval Robustness
Poisoned retrieval benchmark evaluated attack success rate (ASR) with guard off vs on.
- Dataset scenarios: 100
- Baseline ASR: 1.0000 (95% CI: 0.9630, 1.0000)
- Guarded ASR: 0.0000 (95% CI: 0.0000, 0.0370)
- Relative ASR reduction: 1.0000 (95% CI: 1.0000, 1.0000)
- Guarded Safe@1: 1.0000 (95% CI: 0.9630, 1.0000)
Result file: `retrieval/security_guard_benchmark.json`.

## 6. Trust-Spoofed Poisoning Stress Test
A harder variant was executed with poison documents forced to claim high-trust metadata (simulating provenance spoofing).
- Dataset scenarios: 100
- Baseline ASR: 1.0000 (95% CI: 0.9630, 1.0000)
- Guarded ASR: 0.0000 (95% CI: 0.0000, 0.0370)
- Relative ASR reduction: 1.0000 (95% CI: 1.0000, 1.0000)
- Guarded Safe@1: 1.0000 (95% CI: 0.9630, 1.0000)
Result file: `retrieval/security_guard_benchmark_spoofed.json`.
