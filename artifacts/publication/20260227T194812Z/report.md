# Publication Benchmark Artifact Report

- Run ID: `20260227T194812Z`
- Generated: `2026-02-27T19:49:37.891665+00:00`
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
| ttft_ms | 913.649 (402.426, 1889.395) | 222.904 (195.618, 268.052) | 690.745 |
| total_time_ms | 3259.190 (2758.397, 4210.008) | 2361.704 (2128.822, 2526.727) | 897.486 |
| tokens_per_s | 27.626 (27.073, 28.218) | 28.266 (27.816, 28.774) | -0.640 |
| bertscore_f1 | 0.024 (-0.004, 0.049) | 0.136 (0.052, 0.216) | -0.113 |

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
