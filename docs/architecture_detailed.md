# NPC AI Architecture: Dual-Path System

## Overview

This project implements a **Dual-Path Architecture** designed to balance rapid research iteration with high-performance deployment requirements. This hybrid approach allows for experimental flexibility while ensuring the final system meets the strict latency constraints of real-time games (UE5).

## 1. Prototyping Path (Python + Server)
**Focus:** Algorithm Research, Data Generation, Evaluation.

*   **Core:** Python 3.10+
*   **Inference:** Ollama (serving Llama-3/Phi-3 via HTTP)
*   **Orchestration:** `core/inference_adapter.py`
*   **Memory:** Python `ConversationMemory` (Prototypes)
*   **Use Case:**
    *   Evaluating new prompting strategies (e.g., Inner Monologue).
    *   Generating fine-tuning datasets (`LLM-as-a-Judge`).
    *   Testing logic before porting to C++.

## 2. Deployment Path (Native C++ Engine)
**Focus:** Performance, Low Latency, No Dependencies.

*   **Core:** C++17 Standard
*   **Inference:** Custom Inference Engine (ONNX Runtime / LibTorch)
*   **Memory System:** 
    *   **Vector Store:** Custom `VectorStore.cpp` using `usearch` (HNSW) and Half-Float quantization.
    *   **KV Cache:** `KVCacheManager.cpp` for stateful context management across turns.
    *   **Semantic Cache:** `SemanticCache.cpp` for O(1) retrieval of frequent queries.
*   **Integration:** Direct DLL binding to Unreal Engine 5 via `NPCInference` plugin.
*   **Advantages:**
    *   **Zero Network Latency:** Inference runs locally within the game process.
    *   **Memory Efficiency:** Quantized KV cache and embeddings.
    *   **Control:** Fine-grained control over generation tokens and stops.

## Architecture Diagram

```mermaid
graph TD
    subgraph UE5 [Unreal Engine 5 Game Client]
        Player[Player Input]
        Context[Context Extractor]
        Player -->|Input| Bridge
        Context -->|State| Bridge
    end

    subgraph Bridge [Native C++ Bridge]
        API[NPCInference Plugin]
    end

    subgraph CppEngine [High-Performance C++ Engine]
        Tokenizer[Tokenizer (SentencePiece)]
        Prompt[PromptBuilder]
        RAG[GraphRAG & VectorStore]
        KV[KV Cache Manager]
        Model[ONNX/Torch Model]
        
        API --> Prompt
        Prompt --> RAG
        RAG --> Prompt
        Prompt --> Tokenizer
        Tokenizer --> Model
        KV -.-> Model
        Model -->|Logits| Tokenizer
        Tokenizer -->|Text| API
    end

    subgraph PythonPath [Research & Proto Path]
        PyAdapter[Python Adapter]
        Ollama[Ollama Server]
        Eval[BERT/LLM Judge]
        
        API -.->|Dev Mode| PyAdapter
        PyAdapter --> Ollama
        Ollama --> PyAdapter
        PyAdapter --> Eval
    end
```

## Detailed Component Analysis

### A. Custom Vector Store (`cpp/src/VectorStore.cpp`)
Instead of using heavy Python libraries like LangChain or ChromaDB, we implemented a lightweight C++ vector store:
*   **Backend:** `usearch` (Simd-optimized HNSW).
*   **Precision:** FP16 (Half-Float) storage to reduce RAM usage by 50%.
*   **Serialization:** Fast binary dump/load for save games.

### B. KV Cache Manager (`cpp/src/KVCacheManager.cpp`)
To support long conversations without re-processing the entire prompt every turn:
*   **Mechanism:** Caches the Key and Value matrices of the Transformer's attention layers.
*   **Optimization:** Only new tokens (User Input) are processed; history is retrieved from VRAM cache.
*   **Result:** 10x speedup on subsequent turns.

### C. GraphRAG (`cpp/src/SimpleGraph.cpp`)
Enhances vector search with structured knowledge:
*   **Structure:** Directed graph (Subject -> Relation -> Target).
*   **Query:** BFS pathfinding to explain *why* an entity is relevant.
*   **Fusion:** Combines Graph lookup results with Vector Semantic Search results.
