#include "NPCInference.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace NPCInference;

struct GroundTruth {
    std::string query;
    std::string expected_phrase;
};

namespace {

std::string GetArgValue(int argc, char* argv[], const std::string& key, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) {
            return std::string(argv[i + 1]);
        }
    }
    return fallback;
}

} // namespace

int main(int argc, char* argv[]) {
    const char* mock_env = std::getenv("NPC_MOCK_MODE");
    if (mock_env && std::string(mock_env) == "1") {
        std::cout << "Retrieval benchmark skipped: NPC_MOCK_MODE=1 bypasses retrieval path." << std::endl;
        return 0;
    }

    const char* model_env = std::getenv("NPC_MODEL_DIR");
    const std::string model_dir = GetArgValue(argc, argv, "--model-dir", model_env ? model_env : "models/phi3_onnx");

    auto engine = std::make_unique<NPCInferenceEngine>();
    NPCInferenceEngine::InferenceConfig config{};
    config.model_dir = model_dir;
    config.enable_rag = true;

    if (!engine->Initialize(config)) {
        std::cerr << "ERROR: failed to initialize engine. Provide --model-dir or NPC_MODEL_DIR." << std::endl;
        return 1;
    }

    const std::vector<std::string> corpus = {
        "The Dragon King lives in the Dark Mountain cave.",
        "The Elven Queen holds the Crystal Scepter.",
        "Health potions can be bought from the Alchemist for 50 gold.",
        "The war of the five kings ended ten years ago.",
        "Iron swords are effective against goblins but weak against trolls."
    };

    for (const auto& doc : corpus) {
        engine->Remember(doc);
    }

    const std::vector<GroundTruth> test_set = {
        {"Where does the dragon live?", "Dark Mountain cave"},
        {"Who has the Crystal Scepter?", "Crystal Scepter"},
        {"How much is a health potion?", "50 gold"},
        {"What weapons are good against goblins?", "effective against goblins"}
    };

    std::cout << "Starting retrieval quality benchmark..." << std::endl;

    int hit_count = 0;
    for (const auto& test : test_set) {
        nlohmann::json state;
        state["language"] = "en";
        (void)engine->GenerateWithState(test.query, state);

        const std::string retrieved = state.value("memory_context", "");
        const bool hit = retrieved.find(test.expected_phrase) != std::string::npos;
        if (hit) hit_count++;

        std::cout << "Query: " << test.query << " -> " << (hit ? "HIT" : "MISS") << std::endl;
    }

    const double hit_at_1 = test_set.empty() ? 0.0 : (100.0 * static_cast<double>(hit_count) / test_set.size());
    std::cout << "\nResults:" << std::endl;
    std::cout << "Hit@1 (phrase-match proxy): " << hit_at_1 << "%" << std::endl;

    return 0;
}
