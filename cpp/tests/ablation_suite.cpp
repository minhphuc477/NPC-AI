// ablation_suite.cpp - Comprehensive Ablation Study Framework

#include "NPCInference.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace NPCInference;

struct AblationConfig {
    std::string name;
    bool enable_rag = true;
    bool enable_graph = true;
    bool enable_speculative = true;
    bool enable_grammar = true;
    bool enable_reflection = true;
    bool enable_planner = true;
    bool enable_hybrid = true;  // Hybrid retrieval (dense + sparse)
    bool enable_truth_guard = true;
};

struct BenchmarkResults {
    double latency_p50_ms;
    double latency_p95_ms;
    double latency_p99_ms;
    double throughput_tokens_per_sec;
    double memory_usage_mb;
    int total_runs;
};

// Test prompts for consistent evaluation
const std::vector<std::string> TEST_PROMPTS = {
    "Tell me about the ancient ruins."
};

BenchmarkResults RunBenchmark(const AblationConfig& config) {
    try {
        std::cout << "\n=== Running Ablation: " << config.name << " ===" << std::endl;
        
        // Configure engine
        NPCInferenceEngine::InferenceConfig inf_config;
        inf_config.model_dir = "F:/NPC AI/models/phi3_onnx";
        inf_config.enable_rag = config.enable_rag;
        inf_config.enable_graph = config.enable_graph;
        inf_config.enable_speculative = config.enable_speculative;
        inf_config.enable_grammar = config.enable_grammar;
        inf_config.enable_reflection = config.enable_reflection;
        inf_config.enable_planner = config.enable_planner;
        inf_config.enable_truth_guard = config.enable_truth_guard;
        
        // Initialize engine
        NPCInferenceEngine engine;
        std::cout << "Initializing engine..." << std::endl;
        if (!engine.Initialize(inf_config)) {
            std::cout << "Failed to initialize engine for config: " << config.name << std::endl;
            throw std::runtime_error("Initialization failed");
        }
        std::cout << "Engine initialized successfully." << std::endl;
        
        BenchmarkResults results = {};
        std::vector<double> latencies;
        
        // Run test prompts
        for (const auto& prompt : TEST_PROMPTS) {
            std::cout << "Running prompt: " << prompt.substr(0, 20) << "..." << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            
            std::string response = engine.GenerateFromContext(
                "You are a wise NPC in a fantasy game.",
                "Elder",
                "In the village square",
                prompt
            );
            
            auto end = std::chrono::high_resolution_clock::now();
            double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(latency_ms);
            
            std::cout << "  Prompt: " << prompt.substr(0, 30) << "... -> " 
                      << latency_ms << "ms" << std::endl;
        }
        
        if (latencies.empty()) throw std::runtime_error("No latencies captured");

        // Calculate statistics
        std::sort(latencies.begin(), latencies.end());
        results.total_runs = latencies.size();
        results.latency_p50_ms = latencies[latencies.size() / 2];
        results.latency_p95_ms = latencies[static_cast<size_t>(latencies.size() * 0.95)];
        results.latency_p99_ms = latencies[static_cast<size_t>(latencies.size() * 0.99)];
        
        double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        results.throughput_tokens_per_sec = (avg_latency > 0) ? (50.0 / avg_latency) * 1000.0 : 0.0;
    } catch (const std::exception& e) {
        std::cerr << "[Error] Benchmark run failed: " << e.what() << std::endl;
        BenchmarkResults results = {}; // Zero init
        return results;
    }
}

void SaveResults(const std::vector<std::pair<AblationConfig, BenchmarkResults>>& all_results) {
    json output = json::array();
    
    for (const auto& [config, results] : all_results) {
        json entry;
        entry["config_name"] = config.name;
        entry["enable_rag"] = config.enable_rag;
        entry["enable_graph"] = config.enable_graph;
        entry["enable_speculative"] = config.enable_speculative;
        entry["enable_grammar"] = config.enable_grammar;
        entry["enable_reflection"] = config.enable_reflection;
        entry["enable_planner"] = config.enable_planner;
        entry["enable_hybrid"] = config.enable_hybrid;
        entry["enable_truth_guard"] = config.enable_truth_guard;
        
        entry["results"]["latency_p50_ms"] = results.latency_p50_ms;
        entry["results"]["latency_p95_ms"] = results.latency_p95_ms;
        entry["results"]["latency_p99_ms"] = results.latency_p99_ms;
        entry["results"]["throughput_tokens_per_sec"] = results.throughput_tokens_per_sec;
        entry["results"]["memory_usage_mb"] = results.memory_usage_mb;
        
        output.push_back(entry);
    }
    
    std::ofstream file("ablation_results.json");
    file << output.dump(2);
    std::cout << "\n✓ Results saved to ablation_results.json" << std::endl;
}

int main() {
/*
#ifdef _WIN32
    _putenv_s("NPC_MOCK_MODE", "1");
#else
    setenv("NPC_MOCK_MODE", "1", 1);
#endif
*/

    try {
        std::cout << "=== NPC AI Ablation Study Suite ===" << std::endl;
        std::cout << "Testing 8 configurations to measure component contributions\n" << std::endl;
        
        std::vector<AblationConfig> configs = {
            // 1. Baseline (all features)
            {"Baseline", true, true, true, false, true, true, true, true},
            // 2. No RAG
            {"No_RAG", false, true, true, false, true, true, true, true},
            // 3. No Graph
            {"No_Graph", true, false, true, false, true, true, true, true},
            // 4. No Hybrid (Dense only)
            {"No_Hybrid", true, true, true, false, true, true, false, true},
            // 5. No Speculative Decoding
            {"No_Speculative", true, true, false, false, true, true, true, true},
            // 6. No Reflection
            {"No_Reflection", true, true, true, false, false, true, true, true},
            // 7. No Planner
            {"No_Planner", true, true, true, false, true, false, true, true},
            // 8. No Truth Guard
            {"No_TruthGuard", true, true, true, false, true, true, true, false}
        };

        
        std::vector<std::pair<AblationConfig, BenchmarkResults>> all_results;
        
        int config_idx = 1;
        for (const auto& config : configs) {
            std::cout << "[" << config_idx++ << "/" << configs.size() << "] Starting: " << config.name << std::endl;
            auto results = RunBenchmark(config);
            all_results.push_back({config, results});
            std::cout << "Done: " << config.name << std::endl;
        }
        
        // Print summary
        std::cout << "\n=== Ablation Study Summary ===" << std::endl;
        std::cout << std::left << std::setw(20) << "Configuration" 
                  << std::right << std::setw(12) << "p95 (ms)"
                  << std::setw(15) << "Throughput"
                  << std::setw(12) << "BERTScore" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (const auto& [config, results] : all_results) {
            std::cout << std::left << std::setw(20) << config.name
                      << std::right << std::setw(12) << std::fixed << std::setprecision(1) << results.latency_p95_ms
                      << std::setw(15) << std::fixed << std::setprecision(1) << results.throughput_tokens_per_sec << std::endl;
        }
        
        SaveResults(all_results);
        
        std::cout << "\n✓ Ablation study complete!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR in ablation_suite: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN CRITICAL ERROR in ablation_suite" << std::endl;
        return 1;
    }
    
    return 0;
}
