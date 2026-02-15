#include "NPCInference.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

using namespace NPCInference;

int main() {
    auto engine = std::make_unique<NPCInferenceEngine>();
    
    NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "F:/NPC AI/models/phi3_onnx";
    config.enable_rag = true;
    config.enable_graph = true;
    
    if (!engine->Initialize(config)) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return 1;
    }

    std::ofstream report("memory_growth_report.csv");
    report << "Iteration,Memory_MB,Total_Tokens,Operation_Latency_MS" << std::endl;

    std::cout << "Starting Memory Growth & Leak Benchmark (100 iterations)..." << std::endl;
    
    const std::string test_prompt = "Tell me more about the history of the kingdom and the war of the five kings.";
    
    for (int i = 1; i <= 100; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Simulate a long response to grow memory/KV-cache
        engine->Generate(test_prompt);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // Record Memory
        auto metrics = engine->GetProfiler().GetMetrics("total");
        double mem_mb = metrics.current_memory_bytes / (1024.0 * 1024.0);
        
        report << i << "," << std::fixed << std::setprecision(2) << mem_mb << "," 
               << metrics.total_tokens << "," << duration << std::endl;
        
        if (i % 10 == 0) {
            std::cout << "Step " << i << "/100: Memory = " << mem_mb << " MB" << std::endl;
        }
    }

    std::cout << "\nBenchmark Complete. Results saved to 'memory_growth_report.csv'." << std::endl;
    std::cout << "Analyze the CSV to check for linear vs exponential growth or persistent leaks." << std::endl;

    return 0;
}
