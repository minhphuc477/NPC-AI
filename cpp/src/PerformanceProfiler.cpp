#include "PerformanceProfiler.h"
#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace NPCInference {

PerformanceProfiler::PerformanceProfiler()
    : start_time_(std::chrono::high_resolution_clock::now()) {
}

PerformanceProfiler::TimingScope PerformanceProfiler::StartTiming(const std::string& operation) {
    return TimingScope(this, operation);
}

void PerformanceProfiler::RecordLatency(const std::string& operation, double latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    latencies_[operation].push_back(latency_ms);
    latencies_["total"].push_back(latency_ms);
}

void PerformanceProfiler::RecordTokens(size_t token_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_tokens_ += token_count;
}

void PerformanceProfiler::RecordRequest(bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    total_requests_++;
    if (!success) {
        failed_requests_++;
    }
}

void PerformanceProfiler::UpdateMemoryUsage(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_memory_ = bytes;
    if (bytes > peak_memory_) {
        peak_memory_ = bytes;
    }
}

void PerformanceProfiler::CalculatePercentiles(const std::string& operation, Metrics& metrics) const {
    auto it = latencies_.find(operation);
    if (it == latencies_.end() || it->second.empty()) {
        return;
    }

    std::vector<double> sorted = it->second;
    std::sort(sorted.begin(), sorted.end());

    size_t n = sorted.size();
    metrics.min_latency = sorted[0];
    metrics.max_latency = sorted[n - 1];
    metrics.p50_latency = sorted[n / 2];
    metrics.p95_latency = sorted[static_cast<size_t>(n * 0.95)];
    metrics.p99_latency = sorted[static_cast<size_t>(n * 0.99)];

    double sum = 0.0;
    for (double val : sorted) {
        sum += val;
    }
    metrics.mean_latency = sum / n;
}

PerformanceProfiler::Metrics PerformanceProfiler::GetMetrics(const std::string& operation) const {
    std::lock_guard<std::mutex> lock(mutex_);

    Metrics metrics;
    CalculatePercentiles(operation, metrics);

    metrics.total_requests = total_requests_;
    metrics.failed_requests = failed_requests_;
    metrics.total_tokens = total_tokens_;
    metrics.peak_memory_bytes = peak_memory_;
    metrics.current_memory_bytes = current_memory_;

    // Calculate throughput
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
    
    if (elapsed > 0) {
        metrics.requests_per_second = static_cast<double>(total_requests_) / elapsed;
        metrics.tokens_per_second = static_cast<double>(total_tokens_) / elapsed;
    }

    return metrics;
}

void PerformanceProfiler::PrintSummary() const {
    auto metrics = GetMetrics("total");

    std::cerr << "\n=== Performance Metrics ===" << std::endl;
    std::cerr << "Requests: " << metrics.total_requests 
              << " (failed: " << metrics.failed_requests << ")" << std::endl;
    std::cerr << "Tokens: " << metrics.total_tokens << std::endl;
    std::cerr << "\nLatency (ms):" << std::endl;
    std::cerr << "  Mean: " << metrics.mean_latency << std::endl;
    std::cerr << "  P50:  " << metrics.p50_latency << std::endl;
    std::cerr << "  P95:  " << metrics.p95_latency << std::endl;
    std::cerr << "  P99:  " << metrics.p99_latency << std::endl;
    std::cerr << "  Min:  " << metrics.min_latency << std::endl;
    std::cerr << "  Max:  " << metrics.max_latency << std::endl;
    std::cerr << "\nThroughput:" << std::endl;
    std::cerr << "  Requests/sec: " << metrics.requests_per_second << std::endl;
    std::cerr << "  Tokens/sec:   " << metrics.tokens_per_second << std::endl;
    std::cerr << "\nMemory:" << std::endl;
    std::cerr << "  Current: " << (metrics.current_memory_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cerr << "  Peak:    " << (metrics.peak_memory_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cerr << "==========================\n" << std::endl;
}

bool PerformanceProfiler::ExportToJSON(const std::string& filepath) const {
    try {
        std::lock_guard<std::mutex> lock(mutex_);

        json j;
        j["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
        j["total_requests"] = total_requests_;
        j["failed_requests"] = failed_requests_;
        j["total_tokens"] = total_tokens_;
        j["peak_memory_bytes"] = peak_memory_;
        j["current_memory_bytes"] = current_memory_;

        // Export per-operation metrics
        json operations = json::object();
        for (const auto& [op, latencies] : latencies_) {
            Metrics metrics;
            CalculatePercentiles(op, metrics);

            operations[op] = {
                {"mean_latency", metrics.mean_latency},
                {"p50_latency", metrics.p50_latency},
                {"p95_latency", metrics.p95_latency},
                {"p99_latency", metrics.p99_latency},
                {"min_latency", metrics.min_latency},
                {"max_latency", metrics.max_latency},
                {"sample_count", latencies.size()}
            };
        }
        j["operations"] = operations;

        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return false;
        }

        file << j.dump(2);
        file.close();
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error exporting metrics: " << e.what() << std::endl;
        return false;
    }
}

void PerformanceProfiler::Reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    latencies_.clear();
    total_requests_ = 0;
    failed_requests_ = 0;
    total_tokens_ = 0;
    peak_memory_ = 0;
    current_memory_ = 0;
    start_time_ = std::chrono::high_resolution_clock::now();
}

} // namespace NPCInference
