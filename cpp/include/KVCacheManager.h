#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <list>
#include <mutex>
#include <optional>

namespace NPCInference {

/**
 * Persistent KV-Cache Manager with LRU Eviction
 * 
 * Features:
 * - Per-conversation cache persistence
 * - LRU eviction when memory limit reached
 * - Thread-safe operations
 * - Cache statistics tracking
 */
class KVCacheManager {
public:
    struct CacheEntry {
        std::vector<Ort::Value> kv_tensors;
        size_t sequence_length;
        int64_t last_access_time;
        size_t memory_bytes;
    };

    struct CacheStats {
        size_t total_entries = 0;
        size_t total_memory_bytes = 0;
        size_t hits = 0;
        size_t misses = 0;
        size_t evictions = 0;
        
        float hit_rate() const {
            return (hits + misses > 0) ? static_cast<float>(hits) / (hits + misses) : 0.0f;
        }
    };

    /**
     * Constructor
     * @param max_memory_mb Maximum memory in MB for all caches
     * @param max_entries Maximum number of cache entries
     */
    explicit KVCacheManager(size_t max_memory_mb = 512, size_t max_entries = 100);
    ~KVCacheManager();

    /**
     * Get cache for a conversation ID
     * @param conversation_id Unique identifier for conversation
     * @return Optional CacheEntry (Copied for safety)
     */
    std::optional<CacheEntry> Get(const std::string& conversation_id);

    /**
     * Store or update cache for a conversation
     * @param conversation_id Unique identifier
     * @param kv_tensors KV cache tensors (moved)
     * @param sequence_length Current sequence length
     */
    void Put(const std::string& conversation_id, 
             std::vector<Ort::Value>&& kv_tensors,
             size_t sequence_length);

    /**
     * Remove cache for a conversation
     */
    void Remove(const std::string& conversation_id);

    /**
     * Store System Prompt KV separately (Persistent across conversations)
     */
    void PutSystemKV(std::vector<Ort::Value>&& kv_tensors, size_t sequence_length);
    
    /**
     * Retrieve System Prompt KV
     * @return Optional CacheEntry (Copied for safety)
     */
    std::optional<CacheEntry> GetSystemKV();

    /**
     * Clear all caches
     */
    void Clear();

    /**
     * Get cache statistics
     */
    CacheStats GetStats() const;

    /**
     * Set maximum memory limit
     */
    void SetMaxMemory(size_t max_memory_mb);

    /**
     * Deep copy KV Cache tensors
     */
    static std::vector<Ort::Value> CloneKV(const std::vector<Ort::Value>& source);

private:
    void EvictLRU();
    size_t EstimateMemoryUsage(const std::vector<Ort::Value>& tensors);
    void UpdateAccessTime(const std::string& conversation_id);

    // LRU tracking: list of conversation IDs in access order (front = most recent)
    std::list<std::string> lru_list_;
    
    // Map from conversation_id to cache entry and LRU iterator
    std::unordered_map<std::string, std::pair<CacheEntry, std::list<std::string>::iterator>> cache_map_;
    
    // Dedicated System Prompt Cache
    std::unique_ptr<CacheEntry> system_prompt_cache_;

    size_t max_memory_bytes_;
    size_t max_entries_;
    size_t current_memory_bytes_ = 0;
    
    mutable std::mutex mutex_;
    CacheStats stats_;
};

} // namespace NPCInference
