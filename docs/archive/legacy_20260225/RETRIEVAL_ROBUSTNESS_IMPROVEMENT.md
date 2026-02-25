# Retrieval Robustness Improvement (Poisoning / Prompt Injection)

## Why this was chosen
Recent RAG literature repeatedly reports a weakness: retrieved documents can be
poisoned or injection-laden, causing unsafe or incorrect generation.

References:
- PoisonedRAG (2024): https://arxiv.org/abs/2402.07867
- How to Catch an AI Liar (ACL 2024 findings):
  https://aclanthology.org/2024.findings-acl.846/
- Practical Poisoning Attacks on RAG (2025):
  https://arxiv.org/abs/2506.03180
- RAGuard (2025): https://arxiv.org/abs/2504.04712
- RAGForensics (USENIX Security 2026):
  https://www.usenix.org/conference/usenixsecurity26/presentation/xiang-jiuhao

## What was implemented
1. Trust-aware and injection-risk-aware reranking/filtering in `HybridRetriever`.
2. Directive cue scoring for imperative override patterns in retrieved text.
3. Metadata-claim consistency checks to resist trust-spoofed poisoning.
4. New benchmark mode for trust-spoofed poisoned passages.
5. Publication pipeline integration for dual security artifacts (standard + spoofed).
6. New retrieval guard config controls in `NPCInference::InferenceConfig`.
7. Citation payload now exposes robustness features for observability.
8. Memory ingestion now preserves retrieval provenance metadata end-to-end.
9. Added tests and a synthetic adversarial benchmark.

Key files:
- `cpp/include/HybridRetriever.h`
- `cpp/src/HybridRetriever.cpp`
- `cpp/include/NPCInference.h`
- `cpp/src/NPCInference.cpp`
- `cpp/tests/test_retrieval_guard.cpp`
- `cpp/tests/bench_retrieval_security.cpp`

## Measured result
Run artifact:
- `artifacts/publication/20260224T151628Z/retrieval/security_guard_benchmark.json`
- `artifacts/publication/20260224T151628Z/retrieval/security_guard_benchmark_spoofed.json`

Synthetic adversarial benchmark output:
- Baseline attack success rate (ASR): `1.0`
- Guarded ASR: `0.0`
- Relative ASR reduction: `1.0`
- Guarded Safe@1: `1.0`

Trust-spoofed stress output:
- Baseline ASR: `1.0`
- Guarded ASR: `0.0`
- Relative ASR reduction: `1.0`
- Guarded Safe@1: `1.0`

Internal before/after ablation (same spoofed setting):
- Before patch guarded ASR: `0.83`, Safe@1: `0.17`
  (`artifacts/publication/20260224T151557Z/retrieval/security_guard_benchmark_spoofed_before_patch.json`)
- After patch guarded ASR: `0.00`, Safe@1: `1.00`
  (`artifacts/publication/20260224T151628Z/retrieval/security_guard_benchmark_spoofed.json`)

## Reproduce
```bash
cmake --build cpp/build --config Release --target bench_retrieval_security
cpp/build/Release/bench_retrieval_security.exe --output artifacts/publication/20260224T151628Z/retrieval/security_guard_benchmark.json

python scripts/run_publication_benchmark_suite.py --repeats 1 --max-tokens 64 --temperature 0.2 --run-security-benchmark
python scripts/run_publication_benchmark_suite.py --repeats 1 --max-tokens 64 --temperature 0.2 --run-security-benchmark --run-security-spoofed-benchmark
```

## Scope note
This establishes superiority on one targeted weakness axis (retrieval poisoning
robustness). It does not claim overall serving speed superiority.
