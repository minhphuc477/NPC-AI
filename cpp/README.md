# C++ NPC Inference Engine (Core)

This directory contains the high-performance C++ implementation of the BD-NSCA inference engine. It is designed to be embedded directly into Unreal Engine 5 or run as a high-speed standalone service.

### Engine Component Architecture

```text
+-----------------------------------------------------------+
|                  NPC Inference Engine                     |
+--------------------------+--------------------------------+
                           |
          +----------------+----------------+
          |                |                |
   +------v-------+ +------v-------+ +------v-------+
   | Neural Core  | | Knowledge    | | Memory Mgmt   |
   +--------------+ +--------------+ +--------------+
   | ModelLoader  | | HybridRetr.  | | Consolidator  |
   | Tokenizer    | | VectorStore  | | SemanticCache |
   | GrammarSamp. | | SimpleGraph  | | Profiler      |
   +--------------+ +--------------+ +--------------+
```

## Core Components

The engine is modular, allowing for flexible configuration of retrieval and inference strategies.

### Neural Components
- **ModelLoader**: Manages ONNX Runtime sessions. Supports Speculative Decoding for 2x-3x speedups on CPUs.
- **Tokenizer**: Native SentencePiece integration for efficient text encoding/decoding.
- **GrammarSampler**: A logit processor that enforces structural constraints (e.g., JSON schemas) during token selection.

### Retrieval & Knowledge
- **HybridRetriever**: Orchestrates dense and sparse search results using Reciprocal Rank Fusion (RRF).
- **VectorStore**: Dense memory storage utilizing usearch for fast HNSW-based vector similarity.
- **BM25Retriever**: Classic sparse keyword matching for exact fact retrieval.
- **SimpleGraph**: Relational knowledge base for world facts and NPC-Player relationships.

### Memory Management
- **MemoryConsolidator**: Implements importance-based memory pruning and summarization.
- **SemanticCache**: Caches frequent query-response pairs to skip LLM inference for repetitive interactions.

---

## Developer Setup

### Dependencies
The build system uses FetchContent to automatically manage dependencies:
- **ONNX Runtime**: (Auto-downloaded) For neural inference.
- **SentencePiece**: (Auto-downloaded) For tokenization.
- **usearch**: (Auto-downloaded) For vector search.
- **nlohmann/json**: (Auto-downloaded) For data serialization.

### Advanced Build Options
```powershell
cmake -B build `
  -DUSE_CUDA=ON `          # Enable GPU acceleration
  -DBUILD_TESTS=ON `       # Build reliability tests
  -DBUILD_CLI=ON           # Build the standalone npc_cli tool
```

---

## Verification Suite

We use a three-tier testing approach:
1.  **test_fix.exe**: Unit tests for retrieval logic and persistence.
2.  **test_grammar.exe**: Validation of logit masking and structured output.
3.  **test_integration.exe**: End-to-end verification of the Cognitive Architecture loop.

---

## Library Integration

### Basic Inference
```cpp
#include "NPCInference.h"

NPCInference::NPCInferenceEngine engine;
engine.Initialize("models/npc_phi3");

// Concurrent state management
nlohmann::json state = {{"npc_id", "Aria"}, {"mood", "Happy"}};
std::string response = engine.GenerateWithState("Hello Aria!", state);
```

### Memory & State
```cpp
// Add a persistent memory with metadata
engine.Remember("The player found the lost temple.", {{"importance", "high"}});

// Trigger a sleep cycle to consolidate memories
engine.PerformSleepCycle();
```

---

## Performance Profiling
The engine includes a PerformanceProfiler to track latency across different stages (Tokenization, Retrieval, Decoding). Enable profiling in InferenceConfig to see detailed bottlenecks in the console.
