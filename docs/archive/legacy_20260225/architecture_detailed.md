# NPC AI Architecture (Detailed)

## Scope
This file expands [ARCHITECTURE.md](ARCHITECTURE.md) with component contracts and failure behavior.

## Component Contracts
1. Context Extractor (UE5)
- Inputs: actor state, behavior tree state, world events.
- Outputs: normalized context object for prompt assembly.
- Constraint: must degrade gracefully if one signal is missing.

2. Prompt Builder
- Inputs: persona profile, dynamic context, retrieval snippets, player utterance.
- Outputs: bounded prompt string with deterministic section markers.
- Constraint: enforce prompt size limits before model call.

3. Hybrid Retriever
- Inputs: query text and optional state hints.
- Outputs: ranked evidence list with scores and provenance metadata.
- Constraint: apply risk-aware filtering before final ranking.

4. Generation Runtime
- Inputs: prompt, generation params.
- Outputs: raw model text.
- Constraint: timeout and error handling must return structured failure state.

5. Response Controller
- Inputs: raw output + context/persona heuristics.
- Outputs: final response candidate.
- Constraint: sanitize first, then rewrite/select, then grounded fallback.

6. Memory Layer
- Inputs: response text, selected citations, interaction metadata.
- Outputs: append/prune/consolidate operations.
- Constraint: bounded memory growth with periodic consolidation.

## Failure Handling
- Retrieval unavailable: continue with context-only prompt.
- Model unavailable: return deterministic fallback text.
- Tool unavailable: mark tool result as unavailable/simulated, do not crash.
- Unsafe/ungrounded generation: force response-controller fallback.

## C++ and Python Parity
- C++ runtime path: `cpp/src/ResponseController.cpp`.
- Python eval/adapter path: `core/response_controller.py`.
- Expected parity: both paths must execute sanitize -> rewrite/select -> fallback semantics.

## Benchmark Traceability
- Proposal evaluation entrypoint: `scripts/run_proposal_alignment_eval.py`
- Publication evaluation entrypoint: `scripts/run_publication_benchmark_suite.py`
- Artifact outputs: `artifacts/proposal/*`, `artifacts/publication/*`

## Current Gaps
- Serving latency superiority over lightweight external baselines is not yet proven.
- Human-eval coverage depends on final multi-rater annotation completion.
