# NPC AI Benchmark Standards

This document defines measurement standards and reporting rules. It does not
assert final scores unless artifact files are attached.

## Published Artifact Runner

Use `scripts/run_publication_benchmark_suite.py` to generate a timestamped
artifact bundle under `artifacts/publication/<RUN_ID>/` with:
- non-mock serving traces
- hardware/model metadata
- standardized retrieval metrics (Hit@k/MRR/nDCG)
- confidence intervals and baseline deltas
- optional adversarial retrieval security metrics (ASR)
- a machine-generated markdown report

Example:
```bash
python scripts/run_publication_benchmark_suite.py --repeats 1 --max-tokens 64 --temperature 0.2 --run-security-benchmark
```

## 1. Runtime Latency (C++)

Measure with non-mock runs (`NPC_MOCK_MODE=0`) whenever model weights are
available.

| Metric | Reporting Rule | Tool |
|---|---|---|
| Cold start | report median + p95 | `bench_engine` |
| End-to-end generation latency | report p50/p95/p99 | `bench_engine` |
| Retrieval latency | report p50/p95 by query set | `bench_retrieval` |
| Ablation deltas | report relative change vs baseline | `ablation_suite` |

Required metadata:
- hardware (CPU/GPU/RAM)
- model IDs and quantization
- run count
- mock mode on/off

## 2. Retrieval Quality

Use labeled query-doc relevance files and report:
- `Hit@k`
- `MRR`
- `nDCG@k`

The script `scripts/evaluate_benchmarks.py` now supports file-based retrieval
evaluation via `--rag-gold` and `--rag-predictions`.

## 3. Dialogue Quality

Report metrics on fixed test splits:
- semantic similarity/quality (BERT-based evaluators)
- persona consistency
- groundedness (citation-aware checks where available)

If using LLM-as-a-judge, publish prompts and judge model version.

## 4. Reliability

Track over long runs:
- memory growth trends
- error rate / fallback rate
- tool-call availability rate (provider-backed vs unavailable/simulated)
- adversarial retrieval attack success rate (ASR) with guard OFF vs ON

## 5. Claim Discipline

- `Mock-validated` results can validate wiring and control flow.
- `Production-validated` results require non-mock artifacts and metadata.
- Do not claim superiority against literature or industry without published,
  reproducible non-mock benchmark artifacts.
