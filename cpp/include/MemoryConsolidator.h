#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include "Tokenizer.h"

namespace NPCInference {

class ModelLoader;
class Tokenizer; // Forward declare just in case, though header is included.

struct UnconsolidatedMemory {
    std::string text;
    std::string role; // "Player" or "NPC"
    std::map<std::string, std::string> metadata;
};

class MemoryConsolidator {
public:
    MemoryConsolidator(ModelLoader* model_loader, Tokenizer* tokenizer);
    ~MemoryConsolidator();

    /**
     * Summarize a list of conversation turns into a concise paragraph.
     */
    std::string SummarizeConversation(const std::vector<UnconsolidatedMemory>& memories, int max_words = 150);

    /**
     * Extract key facts from a text block.
     */
    std::vector<std::string> ExtractFacts(const std::string& text);

    /**
     * Rate the importance of a memory (0.0 to 1.0).
     */
    /**
     * Rate the importance of a memory (0.0 to 1.0).
     */
    float AssessImportance(const std::string& memory_text);

    /**
     * Generate a high-level insight/reflection based on recent memories.
     * Implements "Generative Agents" Reflection pattern.
     */
    std::string GenerateReflectiveInsight(const std::string& recent_memories);

private:
    ModelLoader* model_loader_;
    Tokenizer* tokenizer_;

    // Helper to run a prompt and get text back
    std::string QueryLLM(const std::string& prompt, int max_tokens = 200);
};

} // namespace NPCInference
