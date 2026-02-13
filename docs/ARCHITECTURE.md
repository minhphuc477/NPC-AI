# NPC AI Architecture

## 1. System Overview

The NPC AI system is a high-performance, local inference engine designed for dynamic non-player characters in Unreal Engine 5 games. It combines a quantized LLM with neuro-symbolic validation, long-term memory, and multi-modal capabilities.

### Core Philosophy
- **Local-First**: Runs entirely on-device (CPU/GPU) without cloud dependencies.
- **Neuro-Symbolic**: Combines LLM creativity with symbolic logic (Truth Guard) for consistency.
- **Asynchronous**: Non-blocking inference pipeline to maintain game frame rates.

---

## 2. Component Architecture

### 2.1 Inference Engine (`NPCInference`)
The central coordinator that manages the lifecycle of a request:
1.  **Prompt Building**: Assembles context, history, and system instructions.
2.  **Retrieval (RAG)**: Fetches relevant memories from Vector Store and Knowledge Graph.
3.  **Generation**: Runs the LLM (Phi-3) with speculative decoding.
4.  **Validation**: Checks output against Truth Guard.
5.  **Execution**: Parses and triggers tool calls or memory updates.

### 2.2 Vision Pipeline (`VisionLoader`)
- **Model**: CLIP ViT-B/32 (INT8) or Phi-3-Vision.
- **Workflow**:
  - Preprocesses image (Resize 336x336, Normalize).
  - Encodes visual features into embeddings.
  - Injects scene description into the prompt context.
- **Performance**: ~20ms latency on CPU (INT8).

### 2.3 Structured Generation (`GrammarSampler`)
- **Mechanism**: Finite State Machine (FSM) constrained sampling.
- **Features**:
  - Enforces strict JSON schema compliance.
  - 13-state automaton tracking (ObjectKey, ObjectValue, Array, etc.).
  - Zero-cost overhead compared to standard sampling.

### 2.4 Memory System
- **Working Memory**: KV Cache managed by `KVCacheManager`.
- **Episodic Memory**: `VectorStore` (usearch) for semantic retrieval.
- **Semantic Memory**: `SimpleGraph` for entity relationships.
- **Consolidation**: `MemoryConsolidator` runs during "sleep cycles" to summarize and prune memories.

---

## 3. Advanced Optimizations

### 3.1 Speculative Decoding
- **Draft Model**: Llama-68M (INT4).
- **Target Model**: Phi-3-Mini (INT4).
- **Method**: Draft model predicts K tokens; Target model verifies in parallel.
- **Speedup**: ~1.7x on CPU.

### 3.2 Flash Attention 2
- **Integration**: `FlashAttention2Integration` module.
- **Benefit**: O(N) memory complexity for attention.
- **Requirement**: CUDA-capable GPU.

### 3.3 Quantization
- **Format**: INT4 (GPTQ/AWQ).
- **Accuracy**: <1% perplexity degradation.
- **Memory**: ~2GB VRAM per 3B parameters.

---

## 4. Hardware Requirements

| Tier | CPU | RAM | GPU | Features |
|------|-----|-----|-----|----------|
| **Minimum** | 4-core | 8GB | Integrated | CPU Inference, No Vision |
| **Recommended** | 8-core | 16GB | RTX 3060 (12GB) | Full RAG, Vision, Tools |
| **High-End** | 12-core | 32GB | RTX 4090 | 4K Vision, Max Context |

---

## 5. Directory Structure

```
NPC AI/
├── cpp/                # Core C++ Inference Engine
│   ├── src/            # Implementation files
│   ├── include/        # Header files
│   └── tests/          # Unit assessments & Benchmarks
├── python/             # Prototype & Training scripts
├── ue5/                # Unreal Engine 5 Plugin
├── models/             # ONNX/GGUF Model files
├── docs/               # Documentation
└── scripts/            # Utilities (Export, Quantize, Benchmark)
```
