// ablation_suite.cpp - Comprehensive ablation benchmark harness.

#include "NPCInference.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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
    bool enable_hybrid = true;
    bool enable_truth_guard = true;
};

struct BenchmarkResults {
    double latency_p50_ms = 0.0;
    double latency_p95_ms = 0.0;
    double latency_p99_ms = 0.0;
    double throughput_tokens_per_sec = 0.0;
    double memory_usage_mb = 0.0;
    int total_runs = 0;
};

struct CliOptions {
    std::string model_dir = "models/phi3_onnx";
    std::string output_path = "ablation_results.json";
    int runs_per_prompt = 3;
    std::string mock_mode;
};

namespace {

const std::vector<std::string> kTestPrompts = {
    "Tell me about the ancient ruins."
};

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

CliOptions ParseArgs(int argc, char* argv[]) {
    CliOptions opts{};
    const char* env_model_dir = std::getenv("NPC_MODEL_DIR");
    opts.model_dir = GetArgValue(argc, argv, "--model-dir", env_model_dir ? env_model_dir : opts.model_dir);
    opts.output_path = GetArgValue(argc, argv, "--output", opts.output_path);
    opts.runs_per_prompt = std::max(1, GetIntArgValue(argc, argv, "--runs", opts.runs_per_prompt));
    opts.mock_mode = GetArgValue(argc, argv, "--mock-mode", "");
    return opts;
}

void ConfigureMockMode(const CliOptions& opts) {
    if (opts.mock_mode.empty()) return;
#ifdef _WIN32
    _putenv_s("NPC_MOCK_MODE", opts.mock_mode.c_str());
#else
    setenv("NPC_MOCK_MODE", opts.mock_mode.c_str(), 1);
#endif
}

BenchmarkResults RunBenchmark(const AblationConfig& config, const CliOptions& opts) {
    BenchmarkResults results{};

    try {
        std::cout << "\n=== Running Ablation: " << config.name << " ===" << std::endl;

        NPCInferenceEngine::InferenceConfig inf_config{};
        inf_config.model_dir = opts.model_dir;
        inf_config.enable_rag = config.enable_rag;
        inf_config.enable_graph = config.enable_graph;
        inf_config.enable_speculative = config.enable_speculative;
        inf_config.enable_grammar = config.enable_grammar;
        inf_config.enable_reflection = config.enable_reflection;
        inf_config.enable_planner = config.enable_planner;
        inf_config.enable_truth_guard = config.enable_truth_guard;

        NPCInferenceEngine engine;
        std::cout << "Initializing engine..." << std::endl;
        if (!engine.Initialize(inf_config)) {
            throw std::runtime_error("Initialization failed for config " + config.name);
        }

        std::vector<double> latencies_ms;
        latencies_ms.reserve(kTestPrompts.size() * static_cast<size_t>(opts.runs_per_prompt));
        size_t total_generated_tokens = 0;

        for (int run = 0; run < opts.runs_per_prompt; ++run) {
            for (const auto& prompt : kTestPrompts) {
                auto start = std::chrono::high_resolution_clock::now();
                const std::string response = engine.GenerateFromContext(
                    "You are a wise NPC in a fantasy game.",
                    "Elder",
                    "In the village square",
                    prompt
                );
                auto end = std::chrono::high_resolution_clock::now();

                const double latency_ms =
                    std::chrono::duration<double, std::milli>(end - start).count();
                latencies_ms.push_back(latency_ms);
                total_generated_tokens += std::max<size_t>(1, response.size() / 4);
                std::cout << "  run " << (run + 1) << ": " << latency_ms << " ms" << std::endl;
            }
        }

        if (latencies_ms.empty()) {
            throw std::runtime_error("No latencies captured");
        }

        std::sort(latencies_ms.begin(), latencies_ms.end());
        results.total_runs = static_cast<int>(latencies_ms.size());
        results.latency_p50_ms = Percentile(latencies_ms, 0.50);
        results.latency_p95_ms = Percentile(latencies_ms, 0.95);
        results.latency_p99_ms = Percentile(latencies_ms, 0.99);

        const double total_latency_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
        results.throughput_tokens_per_sec =
            total_latency_ms > 0.0 ? (static_cast<double>(total_generated_tokens) * 1000.0) / total_latency_ms : 0.0;
        results.memory_usage_mb = static_cast<double>(PerformanceProfiler::GetMemoryUsageMB());
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Benchmark run failed: " << e.what() << std::endl;
    }

    return results;
}

void SaveResults(const std::vector<std::pair<AblationConfig, BenchmarkResults>>& all_results,
                 const std::string& output_path) {
    json output = json::array();

    for (const auto& entry : all_results) {
        const AblationConfig& config = entry.first;
        const BenchmarkResults& results = entry.second;

        json row;
        row["config_name"] = config.name;
        row["enable_rag"] = config.enable_rag;
        row["enable_graph"] = config.enable_graph;
        row["enable_speculative"] = config.enable_speculative;
        row["enable_grammar"] = config.enable_grammar;
        row["enable_reflection"] = config.enable_reflection;
        row["enable_planner"] = config.enable_planner;
        row["enable_hybrid"] = config.enable_hybrid;
        row["enable_truth_guard"] = config.enable_truth_guard;

        row["results"]["latency_p50_ms"] = results.latency_p50_ms;
        row["results"]["latency_p95_ms"] = results.latency_p95_ms;
        row["results"]["latency_p99_ms"] = results.latency_p99_ms;
        row["results"]["throughput_tokens_per_sec"] = results.throughput_tokens_per_sec;
        row["results"]["memory_usage_mb"] = results.memory_usage_mb;
        row["results"]["total_runs"] = results.total_runs;

        output.push_back(row);
    }

    std::ofstream file(output_path);
    file << output.dump(2);
    std::cout << "\nResults saved to " << output_path << std::endl;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        const CliOptions opts = ParseArgs(argc, argv);
        ConfigureMockMode(opts);

        std::cout << "=== NPC AI Ablation Study Suite ===" << std::endl;
        std::cout << "Model dir: " << opts.model_dir << std::endl;
        std::cout << "Runs per prompt: " << opts.runs_per_prompt << std::endl;
        if (!opts.mock_mode.empty()) {
            std::cout << "NPC_MOCK_MODE: " << opts.mock_mode << std::endl;
        }

        const std::vector<AblationConfig> configs = {
            {"Baseline", true, true, true, false, true, true, true, true},
            {"No_RAG", false, true, true, false, true, true, true, true},
            {"No_Graph", true, false, true, false, true, true, true, true},
            {"No_Hybrid", true, true, true, false, true, true, false, true},
            {"No_Speculative", true, true, false, false, true, true, true, true},
            {"No_Reflection", true, true, true, false, false, true, true, true},
            {"No_Planner", true, true, true, false, true, false, true, true},
            {"No_TruthGuard", true, true, true, false, true, true, true, false}
        };

        std::vector<std::pair<AblationConfig, BenchmarkResults>> all_results;
        all_results.reserve(configs.size());

        int config_idx = 1;
        for (const auto& config : configs) {
            std::cout << "[" << config_idx++ << "/" << configs.size() << "] " << config.name << std::endl;
            all_results.push_back({config, RunBenchmark(config, opts)});
        }

        std::cout << "\n=== Ablation Summary ===" << std::endl;
        std::cout << std::left << std::setw(20) << "Configuration"
                  << std::right << std::setw(12) << "p95(ms)"
                  << std::setw(14) << "Tok/s"
                  << std::setw(12) << "Mem(MB)"
                  << std::setw(8) << "Runs"
                  << std::endl;
        std::cout << std::string(66, '-') << std::endl;

        for (const auto& entry : all_results) {
            const AblationConfig& config = entry.first;
            const BenchmarkResults& results = entry.second;
            std::cout << std::left << std::setw(20) << config.name
                      << std::right << std::setw(12) << std::fixed << std::setprecision(1) << results.latency_p95_ms
                      << std::setw(14) << std::fixed << std::setprecision(1) << results.throughput_tokens_per_sec
                      << std::setw(12) << std::fixed << std::setprecision(1) << results.memory_usage_mb
                      << std::setw(8) << results.total_runs
                      << std::endl;
        }

        SaveResults(all_results, opts.output_path);
        std::cout << "\nAblation study complete." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR in ablation_suite: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN CRITICAL ERROR in ablation_suite" << std::endl;
        return 1;
    }
}
