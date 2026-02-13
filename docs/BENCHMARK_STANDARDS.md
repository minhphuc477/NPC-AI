# NPC AI Benchmark Standards Proposal

To ensure the NPC AI system remains performant and high-quality as it evolves, we propose the following benchmarking standards.

## 1. System Performance (C++)
Measure the raw efficiency of the inference engine.

| Operation | Target (p95) | Component |
|-----------|--------------|-----------|
| Tokenization | < 5ms | Tokenizer |
| RAG Retrieval | < 50ms | HybridRetriever |
| Graph Search | < 10ms | SimpleGraph |
| Inference (Draft) | < 20ms | MockModel/Draft |
| Inference (Main) | < 150ms/token | ModelLoader |

**Tool**: `bench_engine.exe` (Generated Results: `benchmark_results.json`)

## 2. Advanced Efficiency
Metrics for optimizing the AI architecture.

| Metric | Target | Description |
|--------|--------|-------------|
| Speculative Accept. | > 60% | Percentage of draft tokens accepted by main model. |

## 3. Long-term Reliability
Ensuring the NPC doesn't "leak" memory or slow down over time.

| Metric | Target | Description |
|--------|--------|-------------|
| Memory Growth | Linear | Vector Store should scale O(N) with dialogue turns. |
| Retrieval Latency | Constant | Memory search should not slow down as more memories are added (using HNSW). |

## 4. Model Quality (Python)
Measure the semantic and persona alignment.

| Metric | Target | Description |
|--------|--------|-------------|
| BERTScore F1 | > 0.70 | Semantic similarity to golden responses. |
| Persona Consist. | > 0.80 | Alignment with NPC traits (Merchant, Guard, etc). |
| Context Relev. | > 0.75 | Accuracy in referencing game state. |

**Tool**: `evaluate/evaluate_bertscore.py`

## 3. Resource Usage
| Metric | Target |
|--------|--------|
| Peak Memory | < 4GB (Model dependent) |
| GPU Utilization | > 80% (During batch generation) |

## Implementation Progress
- [x] **C++ Profiler Instrumentation**: `PerformanceProfiler` integrated into the main generation loop.
- [x] **Automated Benchmark Tool**: `bench_engine.cpp` created for high-granularity timing.
- [x] **Quality Evaluation Pipeline**: `evaluate_bertscore.py` available for semantic validation.

## Next Steps
1.  **Continuous Benchmarking**: Run `bench_engine` as part of the CI/CD pipeline.
2.  **Gold Dataset Expansion**: Expand `test_samples.jsonl` to cover more edge cases in Vietnamese dialogue.
3.  **Real-world Profiling**: Capture performance data from UE5 playtests for real-world p99 analysis.
