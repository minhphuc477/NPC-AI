#include "NPCInference.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

using namespace NPCInference;

void RunBench(NPCInferenceEngine* engine, const std::string& config_name, const std::vector<std::string>& prompts) {
    std::cout << "\n>>> Running Ablation: " << config_name << " ..." << std::endl;
    engine->GetProfiler().Reset();
    
    for (const auto& prompt : prompts) {
        engine->Generate(prompt);
        engine->GetProfiler().RecordRequest(true);
    }
    
    engine->GetProfiler().PrintSummary();
    engine->GetProfiler().ExportToJSON("ablation_" + config_name + ".json");
}

int main() {
    auto engine = std::make_unique<NPCInferenceEngine>();
    
    // Default configs for Mock Benchmarking
    NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "mock_path";
    config.draft_model_dir = "mock_draft";
    
    std::vector<std::string> prompts = {
        "Hello, who are you?",
        "Tell me about the dragon in the cave.",
        "How much for that iron sword?",
        "What is the weather like today?",
        "I need a health potion, quickly!"
    };

    // 1. Baseline (All On)
    config.enable_rag = true;
    config.enable_graph = true;
    config.enable_speculative = true;
    engine->Initialize(config);
    RunBench(engine.get(), "baseline", prompts);

    // 2. Ablated Memory (No RAG/KG)
    config.enable_rag = false;
    config.enable_graph = false;
    engine->Initialize(config);
    RunBench(engine.get(), "no_memory", prompts);

    // 3. Ablated Performance (No Speculative)
    config.enable_rag = true;
    config.enable_graph = true;
    config.enable_speculative = false;
    engine->Initialize(config);
    RunBench(engine.get(), "no_speculative", prompts);

    std::cout << "\nAblation Study Complete. Reports saved to ablation_*.json" << std::endl;
    return 0;
}
