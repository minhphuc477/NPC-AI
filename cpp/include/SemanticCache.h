#pragma once

#include "EmbeddingModel.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <list>
#include <mutex>
#include <chrono>

namespace NPCInference {

/**
 * Semantic Cache for RAG queries
 * 
 * Caches query results based on semantic similarity rather than exact match.
 * Uses embedding-based similarity to find cached results for similar queries.
 */
class SemanticCache {
public:
    struct CacheEntry {
        std::string query;
        std::vector<float> query_embedding;
        std::string result;  // Serialized JSON or text
        int64_t timestamp;
        int64_t ttl_seconds;
        size_t hit_count;
    };

    struct CacheStats {
        size_t total_entries = 0;
        size_t hits = 0;
        size_t misses = 0;
        size_t evictions = 0;
        size_t expired = 0;
        
        float hit_rate() const {
            return (hits + misses > 0) ? static_cast<float>(hits) / (hits + misses) : 0.0f;
        }
    };

    /**
     * Constructor
     * @param embedding_model Model for query embeddings
     * @param similarity_threshold Minimum cosine similarity to consider a cache hit (default: 0.95)
     * @param max_entries Maximum cache entries (default: 1000)
     * @param default_ttl_seconds Default TTL in seconds (default: 3600 = 1 hour)
     */
    explicit SemanticCache(std::shared_ptr<EmbeddingModel> embedding_model,
                          float similarity_threshold = 0.95f,
                          size_t max_entries = 1000,
                          int64_t default_ttl_seconds = 3600);

    /**
     * Get cached result for a query
     * Returns nullptr if no similar query found or entry expired
     */
    const CacheEntry* Get(const std::string& query);

    /**
     * Put query result in cache
     */
    void Put(const std::string& query, const std::string& result, int64_t ttl_seconds = -1);

    /**
     * Clear all cache entries
     */
    void Clear();

    /**
     * Remove expired entries
     */
    void RemoveExpired();

    /**
     * Get cache statistics
     */
    CacheStats GetStats() const;

    /**
     * Set similarity threshold
     */
    void SetSimilarityThreshold(float threshold) { similarity_threshold_ = threshold; }

private:
    // Calculate cosine similarity between embeddings
    float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);

    // Get current timestamp
    int64_t GetCurrentTimestamp();

    // Evict LRU entry
    void EvictLRU();

    std::shared_ptr<EmbeddingModel> embedding_model_;
    float similarity_threshold_;
    size_t max_entries_;
    int64_t default_ttl_seconds_;

    // LRU tracking: list of query strings in access order (front = most recent)
    std::list<std::string> lru_list_;
    
    // Map from query to cache entry and LRU iterator
    std::unordered_map<std::string, std::pair<CacheEntry, std::list<std::string>::iterator>> cache_map_;
    
    mutable std::mutex mutex_;
    CacheStats stats_;
};

} // namespace NPCInference
