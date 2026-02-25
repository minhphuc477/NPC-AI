# BD-NSCA Report

## 1. Objective
BD-NSCA targets real-time NPC dialogue with a hybrid approach:
- neural generation for flexibility,
- symbolic and retrieval constraints for consistency.

## 2. Current Architectural Position
The codebase contains core building blocks for memory, graph context, and structured generation. It is architecturally competitive with recent agent papers at the pattern level, but currently lacks enough published non-mock evidence to claim superior results.

## 3. Concrete Improvement Directions
1. Evidence-first benchmarking: publish reproducible non-mock artifacts.
2. Grounded output checks: enforce and evaluate citation usage from retrieval.
3. Tool realism: provider adapter layer is implemented; next step is integrating production weather/knowledge/memory services.
4. Retrieval scoring: calibrate relevance/recency/importance features with labeled data.

## 4. Suggested Evaluation Pack
- Latency: p50/p95/p99 on fixed hardware.
- Retrieval: Hit@k, MRR, nDCG with labeled query-doc sets.
- Dialogue quality: context relevance, persona consistency, groundedness.
- Failure modes: hallucination rate with and without graph/retrieval constraints.

