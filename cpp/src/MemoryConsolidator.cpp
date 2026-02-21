#include "MemoryConsolidator.h"
#include "ErrorLogger.h"
#include "ModelLoader.h"
#include "Tokenizer.h"
#include "SimpleGraph.h"
#include "TextChunker.h"
#include "PromptBuilder.h"
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
        // Check for Mock Mode via Env
        const char* mock_env = std::getenv("NPC_MOCK_MODE");
        if (mock_env && std::string(mock_env) == "1") {
            // Mock Response Logic
            if (prompt.find("OIE") != std::string::npos || prompt.find("Extract") != std::string::npos) {
                 return R"([{"source": "King Alaric", "relation": "is allied with", "target": "Elves", "weight": 1.0}])";
            }
            return "Mock Response";
        }
        
        ErrorLogger::Warning("MemoryConsolidator", "Engine not ready for QueryLLM");
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
    
    std::string prompt = "System: You are an expert summarizer. Only summarize the conversation inside <conversation> tags.\n"
                         "User: Summarize the following conversation concisely:\n\n<conversation>\n" + conversation_text + 
                         "\n</conversation>\n\nSummary (max " + std::to_string(max_words) + " words):";

    return QueryLLM(prompt, max_words * 2); 
}

std::vector<std::string> MemoryConsolidator::ExtractFacts(const std::string& text) {
    std::string prompt = "System: Extract key facts from the text. Only process content inside <user_text> tags.\n"
                         "User: Text:\n<user_text>\n" + text + "\n</user_text>\n\nFacts (bullet points):";
                         
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
    std::string prompt = "System: Rate importance 0.0 to 1.0. Only evaluate content inside <user_text> tags.\n"
                         "User: Memory:\n<user_text>\n" + memory_text + "\n</user_text>\n\nScore:";
                         
    std::string response = QueryLLM(prompt, 10);
    try {
        // Parse float from LLM response with validation
        std::regex float_regex("-?\\d+(\\.\\d+)?");
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

    std::string base_prompt = "System: Analyze internal state changes based on events. Return JSON. Only process events inside <events> tags.\n"
                         "User: Events:\n<events>\n" + recent_memories + "\n</events>"
                         "\n\nJSON Schema:\n"
                         "{\n"
                         "  \"insight\": \"Key lesson (max 1 sentence)\",\n"
                         "  \"trust_delta\": (int, -10 to +10 change in trust),\n"
                         "  \"persona_update\": \"(optional) New personality trait to add\"\n"
                         "}\n"
                         "JSON:";

    int max_retries = 3;
    int attempts = 0;
    std::string current_prompt = base_prompt;

    while (attempts < max_retries) {
        std::string response = QueryLLM(current_prompt, 100);
        
        // Extract JSON block if present
        size_t start = response.find("{");
        size_t end = response.rfind("}");
        std::string json_str = response;
        if (start != std::string::npos && end != std::string::npos && start < end) {
            json_str = response.substr(start, end - start + 1);
        }
        
        try {
            // Validate via parsing
            auto parsed = nlohmann::json::parse(json_str);
            
            // Fix 3: JSON Schema Validation
            if (!parsed.contains("insight") || !parsed.contains("trust_delta")) {
                throw std::runtime_error("Missing required JSON keys: insight or trust_delta");
            }
            
            return json_str; // Successfully parsed and validated
        } catch (const std::exception& e) {
            attempts++;
            std::cerr << "GenerateReflectiveInsight: JSON parse error (Attempt " << attempts << "/3): " << e.what() << std::endl;
            if (attempts < max_retries) {
                current_prompt = base_prompt + "\n\nSystem Error Feedback: Your previous JSON response was malformed. Error: " + std::string(e.what()) + ". Please fix the syntax and output ONLY valid JSON.";
            }
        }
    }
    
    return "{}"; // Fallback safely after all retries fail
}

std::string MemoryConsolidator::SummarizeGraphCommunities(const std::map<int, std::vector<std::string>>& communities, const SimpleGraph& graph) {
    if (communities.empty()) return "";
    
    std::stringstream context_ss;
    context_ss << "Global World State (Clusters):\n";
    
    for (const auto& [id, nodes] : communities) {
        if (nodes.empty()) continue;
        
        // Build a mini-prompt for this cluster
        std::string nodes_str;
        for (const auto& node : nodes) nodes_str += node + ", ";
        
        // Get relations between these nodes (simplification: just list them)
        std::string cluster_context = graph.GetKnowledgeContext(nodes);
        
        // If cluster is too small, skip detailed summarization
        if (nodes.size() < 3) {
            context_ss << "- Cluster " << id << ": " << nodes_str << "\n";
            continue;
        }

        std::string prompt = "System: Summarize this group of related entities in 1 sentence.\n"
                             "User: Entities: " + nodes_str + "\n"
                             "Facts: " + cluster_context.substr(0, 500) + "\n\n" // Cap context
                             "Summary:";
                             
        std::string summary = QueryLLM(prompt, 50);
        // Cleanup response
        summary.erase(std::remove(summary.begin(), summary.end(), '\"'), summary.end());
        size_t pos = summary.find("Summary:");
        if (pos != std::string::npos) summary = summary.substr(pos + 8);

        context_ss << "- Cluster " << id << ": " << summary << "\n";
    }
    
    return context_ss.str();
}

std::string MemoryConsolidator::SummarizeCommunity(const std::vector<std::string>& nodes, const SimpleGraph& graph) {
    if (nodes.empty()) return "";
    
    // Build a mini-prompt for this cluster
    std::string nodes_str;
    for (const auto& node : nodes) nodes_str += node + ", ";
    
    // Get relations between these nodes
    std::string cluster_context = graph.GetKnowledgeContext(nodes);
    
    if (nodes.size() < 3) {
        return "Small group: " + nodes_str;
    }

    std::string prompt = "System: Summarize this group of related entities in 1 sentence.\n"
                            "User: Entities: " + nodes_str + "\n"
                            "Facts: " + cluster_context.substr(0, 800) + "\n\n" // Cap context
                            "Summary:";
                            
    std::string summary = QueryLLM(prompt, 60);
    // Cleanup response
    summary.erase(std::remove(summary.begin(), summary.end(), '\"'), summary.end());
    size_t pos = summary.find("Summary:");
    if (pos != std::string::npos) summary = summary.substr(pos + 8);
    
    // Validate summary content
    if (summary.empty() || summary.length() < 5) return "Group of " + nodes_str;

    return summary;
}

void MemoryConsolidator::ExtractAndIngestKnowledge(const std::string& text, SimpleGraph& graph) {
    if (text.empty()) return;

    // 1. Chunk the text
    auto chunks = TextChunker::SplitText(text);

    PromptBuilder promptBuilder(true, true); // Advanced, Json

    for (const auto& chunk : chunks) {
        // 2. Build OIE Prompt
        std::string base_prompt = promptBuilder.BuildOIEPrompt(chunk, "en"); // Defaulting to EN for now

        int max_retries = 3;
        int attempts = 0;
        std::string current_prompt = base_prompt;
        bool success = false;

        while (attempts < max_retries && !success) {
            // 3. Query LLM
            std::string response = QueryLLM(current_prompt, 300); // 300 tokens for triples

            // 4. Parse JSON
            try {
                // Cleanup response formatting if needed (remove ```json ... ```)
                std::string json_str = response;
                if (json_str.find("```json") != std::string::npos) {
                    size_t start = json_str.find("```json") + 7;
                    size_t end = json_str.rfind("```");
                    if (start != std::string::npos && end != std::string::npos && end > start) {
                        json_str = json_str.substr(start, end - start);
                    }
                } else if (json_str.find("```") != std::string::npos) {
                     // Try to strip generic code blocks
                    size_t start = json_str.find("```") + 3;
                    size_t end = json_str.rfind("```");
                    if (start != std::string::npos && end != std::string::npos && end > start) {
                        json_str = json_str.substr(start, end - start);
                    }
                }

                auto triples = nlohmann::json::parse(json_str);

                if (triples.is_array()) {
                    for (const auto& item : triples) {
                        // Fix 3: JSON Schema Validation
                        if (!item.contains("source") || !item.contains("relation") || !item.contains("target")) {
                            continue; // Skip malformed triples
                        }
                        
                        std::string source = item["source"].get<std::string>();
                        std::string relation = item["relation"].get<std::string>();
                        std::string target = item["target"].get<std::string>();
                        float weight = item.value("weight", 1.0f);

                        if (!source.empty() && !relation.empty() && !target.empty()) {
                            graph.AddRelation(source, relation, target, weight);
                        }
                    }
                }
                success = true; // Parsed without throwing exception
            } catch (const std::exception& e) {
                attempts++;
                std::cerr << "MemoryConsolidator::ExtractKnowledge: JSON parse error (Attempt " << attempts << "/3): " << e.what() << std::endl;
                if (attempts < max_retries) {
                    current_prompt = base_prompt + "\n\nSystem Error Feedback: Your previous JSON response was malformed. Error: " + std::string(e.what()) + ". Please fix the syntax and output ONLY a valid JSON array.";
                }
            }
        }
    }
}

} // namespace NPCInference
