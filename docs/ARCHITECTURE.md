# NPC AI Architecture

## Scope
This document is the architecture source of truth for:
- runtime behavior (`cpp/`, UE5 integration)
- training/evaluation behavior (`core/`, `scripts/`)
- publication evidence linkage (`artifacts/`)

## System Overview
```mermaid
flowchart LR
    UE[UE5 World State] --> CE[Context Extractor]
    PI[Player Input] --> PB[Prompt Builder]
    CE --> PB
    PB --> HR[Hybrid Retriever]
    HR --> GR[Generation Runtime]
    GR --> RC[Response Controller]
    RC --> OUT[NPC Output]
    OUT --> MEM[Memory Update + Citations]

    subgraph Retrieval
        VS[Vector Store]
        BM[BM25]
        RG[Risk Guard]
        VS --> RG
        BM --> RG
    end
    HR --- Retrieval

    subgraph Runtime
        LM[Local LLM]
        TR[Tool Registry]
        LM --> TR
    end
    GR --- Runtime
```

## Runtime Sequence (UE5 -> Response)
```mermaid
sequenceDiagram
    participant U as UE5
    participant E as NPCInferenceEngine
    participant R as HybridRetriever
    participant L as Local LLM
    participant C as ResponseController

    U->>E: GenerateWithState(player_input, dynamic_context)
    E->>R: Retrieve(context_query)
    R-->>E: top_k + citation metadata + risk signals
    E->>L: Prompt(persona + context + retrieval)
    L-->>E: raw response
    E->>C: sanitize/rewrite/select/fallback
    C-->>E: controlled response
    E-->>U: response + citations
    E->>E: memory append/consolidation
```

## Training/Evaluation Path
```mermaid
flowchart TD
    D[Dataset] --> T[train_qlora.py]
    T --> A[Adapter Output]
    A --> I[inference_adapter.py]
    I --> E[proposal/publication evaluation]
    E --> H[human-eval pack + ratings]
    E --> AR[artifact bundles]
    H --> AR
    AR --> QG[proposal_quality_gate.py]
    QG --> PR[publication-ready claims]
```

## Significant-Improvement Loop
```mermaid
flowchart TD
    HE[Human/LLM Multi-rater CSV] --> PD[build_preference_dataset_from_eval.py]
    PD --> DPO[train_dpo.py]
    DPO --> CAND[Updated Candidate Model]
    RG[retrieval_gold + corpus] --> HN[build_retrieval_hard_negative_set.py]
    HN --> RR[Hard-negative reranker training]
    RR --> HR2[Hybrid Retriever v2]
    CAND --> EV[proposal/publication eval]
    HR2 --> EV
    EV --> GATE[proposal_quality_gate.py]
```

## Key Components
1. Context Extractor
- Pulls behavior state, spatial state, nearby entities, and recent events from UE5.

2. Hybrid Retriever
- Fuses dense + sparse retrieval.
- Applies trust/injection-risk guard before final top-k selection.

3. Prompt Builder
- Formats persona, dynamic context, and selected evidence into a constrained prompt layout.

4. Generation Runtime
- Uses local model serving path (Ollama/native path depending on run configuration).

5. Response Controller
- Sanitizes raw output.
- Optionally rewrites and scores multiple candidates.
- Falls back to grounded safe reply if thresholds are not met.

6. Memory + Citations
- Stores response-linked evidence with bounded context payload.

7. Quality Gate
- Enforces proposal/publication quality-bar checks over generated artifacts.
- Verifies scenario coverage, significance wins, metadata completeness, and security metrics.

## Runtime Invariants
- Every response path must support no-crash fallback behavior.
- Retrieval payload must be bounded to avoid prompt explosion.
- Controlled output must preserve persona/context constraints.
- Any superiority claim must map to artifact-backed metrics.

## Unified Checkout
- End-to-end reproducible run (Kaggle/local): `scripts/run_kaggle_full_results.py`
- Produces a single manifest: `artifacts/final_checkout/<timestamp>/manifest.json`

## Primary File Map
- `cpp/src/NPCInference.cpp`
- `cpp/src/HybridRetriever.cpp`
- `cpp/src/ResponseController.cpp`
- `core/response_controller.py`
- `scripts/inference_adapter.py`
- `scripts/run_proposal_alignment_eval.py`
- `scripts/run_publication_benchmark_suite.py`
- `scripts/run_kaggle_full_results.py`
- `scripts/build_human_eval_pack.py`
- `scripts/run_llm_multirater_campaign.py`
- `scripts/build_preference_dataset_from_eval.py`
- `scripts/build_retrieval_hard_negative_set.py`
- `scripts/proposal_quality_gate.py`
