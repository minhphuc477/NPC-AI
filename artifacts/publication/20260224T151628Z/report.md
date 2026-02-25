# Publication Benchmark Artifact Report

- Run ID: `20260224T151628Z`
- Generated: `2026-02-24T15:18:52.632294+00:00`
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

## 3. Confidence Intervals And Ablation Deltas
| Metric | Candidate Mean (95% CI) | Baseline Mean (95% CI) | Delta |
|---|---:|---:|---:|
| ttft_ms | 1604.284 (1007.787, 2746.173) | 414.061 (299.972, 629.916) | 1190.223 |
| total_time_ms | 7016.328 (6288.227, 8245.803) | 5385.800 (4476.079, 5993.042) | 1630.528 |
| tokens_per_s | 11.896 (11.553, 12.259) | 12.009 (11.664, 12.294) | -0.113 |
| bertscore_f1 | 0.023 (-0.013, 0.065) | 0.111 (0.059, 0.160) | -0.088 |

Retrieval ablation deltas (vs BM25 baseline):
| Method | Metric | Absolute Delta | Relative Delta |
|---|---|---:|---:|
| keyword_overlap | hit@5 | -0.1000 | -0.1000 |
| keyword_overlap | mrr | -0.0167 | -0.0196 |
| keyword_overlap | ndcg@5 | -0.0393 | -0.0442 |
| random | hit@5 | -0.6000 | -0.6000 |
| random | mrr | -0.6850 | -0.8059 |
| random | ndcg@5 | -0.6688 | -0.7521 |

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
