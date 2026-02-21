#include "NPCInference.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

using namespace NPCInference;

struct GroundTruth {
    std::string query;
    std::unordered_set<std::string> expected_doc_ids;
};

int main() {
    auto engine = std::make_unique<NPCInferenceEngine>();
    
    NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "mock_path";
    config.enable_rag = true;
    
    if (!engine->Initialize(config)) return 1;

    // 1. Setup Ground Truth Knowledge
    std::vector<std::pair<std::string, std::string>> corpus = {
        {"doc1", "The Dragon King lives in the Dark Mountain cave."},
        {"doc2", "The Elven Queen holds the Crystal Scepter."},
        {"doc3", "Health potions can be bought from the Alchemist for 50 gold."},
        {"doc4", "The war of the five kings ended ten years ago."},
        {"doc5", "Iron swords are effective against goblins but weak against trolls."}
    };

    for (const auto& doc : corpus) {
        engine->Remember(doc.second); // Assuming Remember uses the doc content as label for now
    }

    std::vector<GroundTruth> test_set = {
        {"Where does the dragon live?", {"doc1"}},
        {"Who has the Crystal Scepter?", {"doc2"}},
        {"How much is a health potion?", {"doc3"}},
        {"What weapons are good against goblins?", {"doc5"}}
    };

    std::cout << "Starting Retrieval Quality Benchmark (mAP / Hit@K)..." << std::endl;

    double total_precision = 0;
    int hits_at_1 = 0;
    int hits_at_3 = 0;

    for (const auto& test : test_set) {
        // We need to access retrieved context. 
        // Generate with a mock logic that returns the state.
        nlohmann::json state;
        state["language"] = "en";
        engine->GenerateWithState(test.query, state);
        
        std::string retrieved = state.value("memory_context", "");
        
        // Simple verification: check if expected strings are in the context
        // This is a proxy since we don't have direct access to doc IDs in the current public API
        int match_count = 0;
        for (const auto& doc_id : test.expected_doc_ids) {
            // Find corresponding content
            for (const auto& c : corpus) {
                if (c.first == doc_id && retrieved.find(c.second.substr(0, 20)) != std::string::npos) {
                    match_count++;
                }
            }
        }

        double precision = (double)match_count / 1.0; // Top-1 precision
        total_precision += precision;
        if (match_count > 0) {
            hits_at_1++;
            hits_at_3++; // For top-1, it's also top-3
        }

        std::cout << "Query: " << test.query << " -> " << (match_count > 0 ? "PASS" : "FAIL") << std::endl;
    }

    std::cout << "\nResults:" << std::endl;
    std::cout << "mAP@1: " << (total_precision / test_set.size()) << std::endl;
    std::cout << "Hit@1: " << (double)hits_at_1 / test_set.size() * 100 << "%" << std::endl;

    return 0;
}
