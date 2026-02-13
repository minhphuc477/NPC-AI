#include "MemoryConsolidator.h"
#include "ModelLoader.h"
#include "Tokenizer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>

namespace NPCInference {


MemoryConsolidator::MemoryConsolidator(ModelLoader* model_loader, Tokenizer* tokenizer) 
    : model_loader_(model_loader), tokenizer_(tokenizer) {}

MemoryConsolidator::~MemoryConsolidator() {}

std::string MemoryConsolidator::QueryLLM(const std::string& prompt, int max_tokens) {
    if (!model_loader_ || !model_loader_->IsLoaded() || !tokenizer_ || !tokenizer_->IsLoaded()) {
        std::cerr << "MemoryConsolidator: Engine not ready." << std::endl;
        return "";
    }

    // 1. Encode
    std::vector<int64_t> input_ids = tokenizer_->Encode(prompt);
    if (input_ids.empty()) return "";

    std::vector<int64_t> attention_mask(input_ids.size(), 1);

    // 2. Generate (No cache persistence needed for one-off tasks, use "sys_consolidate")
    std::vector<int64_t> output_ids = model_loader_->Generate(
        input_ids,
        attention_mask,
        max_tokens,
        "sys_consolidate", // Separate cache context
        {},
        {}
    );

    // 3. Decode
    if (output_ids.size() > input_ids.size()) {
        std::vector<int64_t> new_tokens(output_ids.begin() + input_ids.size(), output_ids.end());
        return tokenizer_->Decode(new_tokens);
    }

    return "";
}

std::string MemoryConsolidator::SummarizeConversation(const std::vector<UnconsolidatedMemory>& memories, int max_words) {
    if (memories.empty()) return "";

    std::stringstream ss;
    for (const auto& mem : memories) {
        ss << mem.role << ": " << mem.text << "\n";
    }
    
    std::string conversation_text = ss.str();
    
    std::string prompt = "System: You are an expert summarizer.\n"
                         "User: Summarize the following conversation concisely:\n\n" + conversation_text + 
                         "\n\nSummary (max " + std::to_string(max_words) + " words):";

    return QueryLLM(prompt, max_words * 2); 
}

std::vector<std::string> MemoryConsolidator::ExtractFacts(const std::string& text) {
    std::string prompt = "System: Extract key facts from the text.\n"
                         "User: Text: " + text + "\n\nFacts (bullet points):";
                         
    std::string response = QueryLLM(prompt, 200);
    
    std::vector<std::string> facts;
    std::stringstream ss(response);
    std::string line;
    while (std::getline(ss, line)) {
        if (line.size() > 5 && (line[0] == '-' || line[0] == '*')) {
            facts.push_back(line.substr(1)); // Remove bullet
        }
    }
    return facts;
}

float MemoryConsolidator::AssessImportance(const std::string& memory_text) {
    std::string prompt = "System: Rate importance 0.0 to 1.0.\n"
                         "User: Memory: " + memory_text + "\n\nScore:";
                         
    std::string response = QueryLLM(prompt, 10);
    try {
        // Parse float from LLM response with validation
        std::regex float_regex("[0-9]*\\.[0-9]+");
        std::smatch match;
        if (std::regex_search(response, match, float_regex)) {
            float score = std::stof(match.str());
            
            // Validate range [0.0, 1.0]
            if (score >= 0.0f && score <= 1.0f) {
                return score;
            }
            
            // Out of range - warn and use default
            std::cerr << "MemoryConsolidator: Score out of range [0,1]: " << score 
                      << " (response: '" << response << "'). Using default." << std::endl;
        } else {
            std::cerr << "MemoryConsolidator: No valid float in response: '" 
                      << response << "'. Using default." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "MemoryConsolidator: Parse error: " 
                  << e.what() << ". Using default." << std::endl;
    }
    return 0.5f; // Default
}

std::string MemoryConsolidator::GenerateReflectiveInsight(const std::string& recent_memories) {
    if (recent_memories.empty()) return "";

    std::string prompt = "System: You are an introspective AI. Analyze the recent events and generate a deep insight.\n"
                         "User: Based on these recent events:\n" + recent_memories + 
                         "\n\nWhat is one key insight, belief, or lesson I should remember about my life, relationships, or the world? (Max 1 sentence)\n"
                         "Insight:";

    std::string insight = QueryLLM(prompt, 50);
    
    // Cleanup if LLM chatters
    size_t pos = insight.find("Insight:");
    if (pos != std::string::npos) insight = insight.substr(pos + 8);
    
    // Remove quotes
    insight.erase(std::remove(insight.begin(), insight.end(), '\"'), insight.end());
    
    return insight;
}

} // namespace NPCInference
