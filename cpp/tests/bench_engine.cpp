#include "NPCInference.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <string>

using namespace NPCInference;

namespace {

std::string GetArgValue(int argc, char* argv[], const std::string& key, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) {
            return std::string(argv[i + 1]);
        }
    }
    return fallback;
}

int GetIntArgValue(int argc, char* argv[], const std::string& key, int fallback) {
    const std::string value = GetArgValue(argc, argv, key, "");
    if (value.empty()) return fallback;
    try {
        return std::stoi(value);
    } catch (...) {
        return fallback;
    }
}

double Percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const size_t idx = static_cast<size_t>(std::min<double>(sorted.size() - 1, p * (sorted.size() - 1)));
    return sorted[idx];
}

} // namespace

int main(int argc, char* argv[]) {
    std::cout << "=== NPC AI Performance Benchmark ===" << std::endl;

    const char* model_env = std::getenv("NPC_MODEL_DIR");
    const std::string model_dir = GetArgValue(argc, argv, "--model-dir", model_env ? model_env : "models/phi3_onnx");
    const std::string output_path = GetArgValue(argc, argv, "--output", "benchmark_results.json");
    const int runs = std::max(1, GetIntArgValue(argc, argv, "--runs", 3));

    NPCInferenceEngine::InferenceConfig config{};
    config.model_dir = model_dir;

    auto engine = std::make_unique<NPCInferenceEngine>();

    std::cout << "Initializing engine with model_dir: " << config.model_dir << std::endl;
    if (!engine->Initialize(config)) {
        std::cerr << "ERROR: engine initialization failed. Set --model-dir or NPC_MODEL_DIR to a valid model folder." << std::endl;
        return 1;
    }
    
    // Testing Public API Performance with Profiler
    std::vector<std::string> prompts = {
        "Hello, who are you?",
        "Tell me about the dragon in the cave.",
        "How much for that iron sword?",
        "What is the weather like today?",
        "I need a health potion, quickly!"
    };

    std::cout << "Running " << prompts.size() * runs << " iterations..." << std::endl;
    std::vector<double> latencies_ms;
    latencies_ms.reserve(prompts.size() * static_cast<size_t>(runs));

    for (int run = 0; run < runs; ++run) {
        for (const auto& prompt : prompts) {
            auto t0 = std::chrono::high_resolution_clock::now();
            const std::string response = engine->GenerateFromContext(
                "You are a merchant NPC in a fantasy game.",
                "Elara",
                "Market square",
                prompt
            );
            auto t1 = std::chrono::high_resolution_clock::now();

            const double latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            latencies_ms.push_back(latency_ms);
            engine->GetProfiler().RecordRequest(response.rfind("Error:", 0) != 0);
            engine->GetProfiler().RecordLatency("bench_end_to_end", latency_ms);
        }
    }

    std::sort(latencies_ms.begin(), latencies_ms.end());
    const double p50 = Percentile(latencies_ms, 0.50);
    const double p95 = Percentile(latencies_ms, 0.95);
    const double p99 = Percentile(latencies_ms, 0.99);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Latency summary (ms): p50=" << p50 << ", p95=" << p95 << ", p99=" << p99 << std::endl;

    // Export internal profiler metrics
    engine->GetProfiler().PrintSummary();
    if (!engine->GetProfiler().ExportToJSON(output_path)) {
        std::cerr << "WARNING: failed to export benchmark JSON to " << output_path << std::endl;
    }
    
    std::cout << "\nBenchmark complete. Results saved to " << output_path << std::endl;
    return 0;
}
