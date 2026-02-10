#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <fstream>

namespace NPCInference {

/**
 * Performance Profiler for NPC Inference
 * 
 * Tracks latency, throughput, and resource usage
 */
class PerformanceProfiler {
public:
    struct Metrics {
        // Latency metrics (milliseconds)
        double mean_latency = 0.0;
        double p50_latency = 0.0;
        double p95_latency = 0.0;
        double p99_latency = 0.0;
        double min_latency = 0.0;
        double max_latency = 0.0;
        
        // Throughput
        double tokens_per_second = 0.0;
        double requests_per_second = 0.0;
        
        // Counts
        size_t total_requests = 0;
        size_t total_tokens = 0;
        size_t failed_requests = 0;
        
        // Memory (bytes)
        size_t peak_memory_bytes = 0;
        size_t current_memory_bytes = 0;
    };

    struct TimingScope {
        PerformanceProfiler* profiler;
        std::string operation;
        std::chrono::high_resolution_clock::time_point start;
        
        TimingScope(PerformanceProfiler* p, const std::string& op)
            : profiler(p), operation(op), start(std::chrono::high_resolution_clock::now()) {}
        
        ~TimingScope() {
            if (profiler) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                profiler->RecordLatency(operation, duration.count() / 1000.0);  // Convert to ms
            }
        }
    };

    PerformanceProfiler();

    /**
     * Start timing an operation
     */
    TimingScope StartTiming(const std::string& operation);

    /**
     * Record latency manually
     */
    void RecordLatency(const std::string& operation, double latency_ms);

    /**
     * Record token generation
     */
    void RecordTokens(size_t token_count);

    /**
     * Record request completion
     */
    void RecordRequest(bool success = true);

    /**
     * Update memory usage
     */
    void UpdateMemoryUsage(size_t bytes);

    /**
     * Get metrics for an operation
     */
    Metrics GetMetrics(const std::string& operation = "total") const;

    /**
     * Print metrics summary
     */
    void PrintSummary() const;

    /**
     * Export metrics to JSON file
     */
    bool ExportToJSON(const std::string& filepath) const;

    /**
     * Reset all metrics
     */
    void Reset();

private:
    void CalculatePercentiles(const std::string& operation, Metrics& metrics) const;

    mutable std::mutex mutex_;
    
    // Per-operation latencies
    std::unordered_map<std::string, std::vector<double>> latencies_;
    
    // Global counters
    size_t total_requests_ = 0;
    size_t failed_requests_ = 0;
    size_t total_tokens_ = 0;
    size_t peak_memory_ = 0;
    size_t current_memory_ = 0;
    
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace NPCInference
