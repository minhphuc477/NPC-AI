#pragma once

// NPC AI Constants
// This file contains all magic numbers extracted from the codebase

namespace NPCInference {
namespace Constants {

// Token IDs
constexpr int64_t PHI3_EOS_TOKEN = 32000;
constexpr int64_t PHI3_END_TOKEN = 32007;

// Cache Configuration
constexpr size_t DEFAULT_CACHE_SIZE_MB = 512;
constexpr size_t MAX_CONVERSATIONS = 100;

// Generation Limits
constexpr int DEFAULT_MAX_TOKENS = 150;
constexpr int PLANNING_MAX_TOKENS = 100;
constexpr int CRITIQUE_MAX_TOKENS = 150;
constexpr int REFINE_MAX_TOKENS = 150;
constexpr int TRUTH_CHECK_MAX_TOKENS = 100;

// RAG Configuration
constexpr int DEFAULT_RAG_TOP_K = 3;
constexpr int HYBRID_RAG_TOP_K = 5;
constexpr float DEFAULT_RAG_THRESHOLD = 0.7f;

// Memory Configuration
constexpr float MEMORY_IMPORTANCE_THRESHOLD = 0.3f;
constexpr size_t MIN_MEMORIES_FOR_CONSOLIDATION = 5;
constexpr size_t CONVERSATION_HISTORY_LIMIT = 6;

// Embedding Dimensions
constexpr size_t MINILM_EMBEDDING_DIM = 384;

// Input Validation
constexpr size_t MAX_INPUT_LENGTH = 1000;
constexpr size_t MAX_PROMPT_LENGTH = 4096;

// Timeouts (milliseconds)
constexpr int OLLAMA_TIMEOUT_MS = 30000;
constexpr int MOCK_LATENCY_MS = 50;

} // namespace Constants
} // namespace NPCInference
