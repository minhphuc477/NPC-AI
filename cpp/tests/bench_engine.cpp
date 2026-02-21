#include "NPCInference.h"
#include "ModelLoader.h"
#include "Tokenizer.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace NPCInference;

// Simple Mock for Benchmarking
class BenchModelLoader : public ModelLoader {
public:
    bool IsLoaded() const override { return true; }
    std::vector<int64_t> Generate(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>&,
        int max_tokens,
        const std::string&,
        std::function<void(int64_t)> callback,
        std::function<void(float*, int64_t)>
    ) override {
        // Simulate processing time
        // approx 10ms per token
        std::vector<int64_t> output = input_ids;
        for(int i = 0; i < max_tokens; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            output.push_back(100 + i);
            if (callback) callback(100 + i);
        }
        return output;
    }
};

class BenchTokenizer : public Tokenizer {
public:
    bool IsLoaded() const override { return true; }
    std::vector<int64_t> Encode(const std::string& text) override {
        return std::vector<int64_t>(text.length() / 4 + 1, 1);
    }
    std::string Decode(const std::vector<int64_t>& ids) override {
        return "Benchmarked response token sequence.";
    }
};

int main() {
    std::cout << "=== NPC AI Performance Benchmark ===" << std::endl;

    NPCInferenceEngine::InferenceConfig config;
    auto engine = std::make_unique<NPCInferenceEngine>();
    
    // Wire up for benchmarking
    // In a real scenario, we'd load actual models, but for logic benchmark we use mocks
    // Note: npc_inference lib needs to be compiled with internal access or we test public API
    
    // Testing Public API Performance with Profiler
    std::vector<std::string> prompts = {
        "Hello, who are you?",
        "Tell me about the dragon in the cave.",
        "How much for that iron sword?",
        "What is the weather like today?",
        "I need a health potion, quickly!"
    };

    std::cout << "Running " << prompts.size() << " iterations..." << std::endl;

    for (const auto& prompt : prompts) {
        std::string response = engine->Generate(prompt);
        engine->GetProfiler().RecordRequest(true);
    }

    // Export internal profiler metrics
    engine->GetProfiler().PrintSummary();
    engine->GetProfiler().ExportToJSON("benchmark_results.json");
    
    std::cout << "\nBenchmark Complete. Results saved to benchmark_results.json" << std::endl;
    return 0;
}
