# Industry and Publication Judgment

## Scope
Assess whether the current codebase demonstrates defensible improvement over public baselines.
Use `scripts/proposal_quality_gate.py` for strict pass/fail validation before claiming publication quality.

## Reference Baselines
- Generative Agents: https://arxiv.org/abs/2304.03442
- Speculative Decoding: https://arxiv.org/abs/2211.17192
- GraphRAG: https://arxiv.org/abs/2404.16130
- vLLM speculative decoding docs: https://docs.vllm.ai/en/latest/features/spec_decode.html

## Current Verdict
1. Comparable architecture in memory/retrieval/generation orchestration.
2. Defensible superiority on two targeted axes:
- adversarial retrieval robustness (poisoning resistance)
- context-grounded dialogue quality under response control
3. Not yet superior on raw serving latency/throughput versus lighter baseline models.

## Evidence (Publication Run `20260224T151628Z`)
Artifact root:
- `artifacts/publication/20260224T151628Z`

Completed evidence requirements:
1. Non-mock artifacts with hardware/model metadata.
2. Standard retrieval metrics (Hit@k/MRR/nDCG) on labeled data.
3. Confidence intervals and ablation deltas vs fixed baselines.
4. External baseline comparison on identical prompts/datasets.

Key files:
- `metadata/hardware.json`
- `metadata/models.json`
- `retrieval/metrics.json`
- `retrieval/ablation_deltas_vs_bm25.json`
- `serving/summary.json`
- `serving/delta_vs_baseline.json`

## Superiority Axis 1: Retrieval Poisoning Robustness
Implemented guard:
- trust-aware reranking
- injection-risk penalty
- directive cue checks
- metadata-claim consistency checks

Code:
- `cpp/include/HybridRetriever.h`
- `cpp/src/HybridRetriever.cpp`
- `cpp/tests/test_retrieval_guard.cpp`
- `cpp/tests/bench_retrieval_security.cpp`

Large poisoned benchmark result (100 scenarios):
- baseline ASR: `1.0` (95% CI `[0.9630, 1.0000]`)
- guarded ASR: `0.0` (95% CI `[0.0000, 0.0370]`)
- relative ASR reduction: `1.0`

## Superiority Axis 2: Context-Grounded Dialogue Control
Implemented:
- response-control path in Python + C++ runtime

Code:
- `core/response_controller.py`
- `cpp/src/ResponseController.cpp`
- `scripts/inference_adapter.py`
- `scripts/run_proposal_alignment_eval.py`

Proposal artifact run:
- `artifacts/proposal/20260224T175344Z`

Result summary:
- controlled arm significantly improves context/persona/naturalness/overall quality over same-model no-context arms.
- significant wins also observed against two external no-context baselines on most reported metrics.

## Remaining Gap
- Serving efficiency superiority claim is still unsupported in this evidence set.

## Reproduce
```bash
python scripts/run_publication_benchmark_suite.py --repeats 1 --max-tokens 64 --temperature 0.2 --run-security-benchmark --run-security-spoofed-benchmark
```

Unified full checkout (proposal + publication + human-eval attach + serving matrix + profile suite + gate):
```bash
python scripts/run_kaggle_full_results.py --host http://127.0.0.1:11434
```
