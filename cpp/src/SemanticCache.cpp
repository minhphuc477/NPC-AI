#include "SemanticCache.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace NPCInference {

SemanticCache::SemanticCache(std::shared_ptr<EmbeddingModel> embedding_model,
                             float similarity_threshold,
                             size_t max_entries,
                             int64_t default_ttl_seconds)
    : embedding_model_(embedding_model)
    , similarity_threshold_(similarity_threshold)
    , max_entries_(max_entries)
    , default_ttl_seconds_(default_ttl_seconds) {
}

int64_t SemanticCache::GetCurrentTimestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

float SemanticCache::CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }

    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }

    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

const SemanticCache::CacheEntry* SemanticCache::Get(const std::string& query) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!embedding_model_) {
        return nullptr;
    }

    // Generate query embedding
    std::vector<float> query_embedding = embedding_model_->Embed(query);
    if (query_embedding.empty()) {
        return nullptr;
    }

    // Find most similar cached query
    float best_similarity = 0.0f;
    std::string best_query;

    int64_t current_time = GetCurrentTimestamp();

    for (auto& [cached_query, entry_pair] : cache_map_) {
        auto& entry = entry_pair.first;

        // Check if expired
        if (entry.ttl_seconds > 0 && 
            (current_time - entry.timestamp) > entry.ttl_seconds) {
            continue;  // Skip expired entries
        }

        float similarity = CosineSimilarity(query_embedding, entry.query_embedding);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_query = cached_query;
        }
    }

    // Check if best match exceeds threshold
    if (best_similarity >= similarity_threshold_) {
        stats_.hits++;

        // Update LRU
        auto it = cache_map_.find(best_query);
        if (it != cache_map_.end()) {
            lru_list_.erase(it->second.second);
            lru_list_.push_front(best_query);
            it->second.second = lru_list_.begin();
            it->second.first.hit_count++;

            std::cerr << "SemanticCache: HIT (similarity=" << best_similarity 
                      << ", query='" << query << "' -> cached='" << best_query << "')" << std::endl;

            return &(it->second.first);
        }
    }

    stats_.misses++;
    std::cerr << "SemanticCache: MISS (query='" << query << "')" << std::endl;
    return nullptr;
}

void SemanticCache::Put(const std::string& query, const std::string& result, int64_t ttl_seconds) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!embedding_model_) {
        return;
    }

    // Use default TTL if not specified
    if (ttl_seconds < 0) {
        ttl_seconds = default_ttl_seconds_;
    }

    // Generate query embedding
    std::vector<float> query_embedding = embedding_model_->Embed(query);
    if (query_embedding.empty()) {
        return;
    }

    // Check if query already exists
    auto it = cache_map_.find(query);
    if (it != cache_map_.end()) {
        // Update existing entry
        it->second.first.result = result;
        it->second.first.timestamp = GetCurrentTimestamp();
        it->second.first.ttl_seconds = ttl_seconds;

        // Move to front of LRU
        lru_list_.erase(it->second.second);
        lru_list_.push_front(query);
        it->second.second = lru_list_.begin();

        return;
    }

    // Evict if at capacity
    while (cache_map_.size() >= max_entries_) {
        EvictLRU();
    }

    // Create new entry
    CacheEntry entry;
    entry.query = query;
    entry.query_embedding = std::move(query_embedding);
    entry.result = result;
    entry.timestamp = GetCurrentTimestamp();
    entry.ttl_seconds = ttl_seconds;
    entry.hit_count = 0;

    // Add to LRU front
    lru_list_.push_front(query);
    auto lru_it = lru_list_.begin();

    cache_map_[query] = {std::move(entry), lru_it};
    stats_.total_entries = cache_map_.size();

    std::cerr << "SemanticCache: PUT (query='" << query << "', ttl=" << ttl_seconds << "s)" << std::endl;
}

void SemanticCache::EvictLRU() {
    if (lru_list_.empty()) return;

    // Remove least recently used (back of list)
    std::string victim = lru_list_.back();
    lru_list_.pop_back();

    cache_map_.erase(victim);
    stats_.evictions++;
    stats_.total_entries = cache_map_.size();

    std::cerr << "SemanticCache: Evicted '" << victim << "'" << std::endl;
}

void SemanticCache::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_map_.clear();
    lru_list_.clear();
    stats_.total_entries = 0;
}

void SemanticCache::RemoveExpired() {
    std::lock_guard<std::mutex> lock(mutex_);

    int64_t current_time = GetCurrentTimestamp();
    std::vector<std::string> to_remove;

    for (const auto& [query, entry_pair] : cache_map_) {
        const auto& entry = entry_pair.first;
        if (entry.ttl_seconds > 0 && 
            (current_time - entry.timestamp) > entry.ttl_seconds) {
            to_remove.push_back(query);
        }
    }

    for (const auto& query : to_remove) {
        auto it = cache_map_.find(query);
        if (it != cache_map_.end()) {
            lru_list_.erase(it->second.second);
            cache_map_.erase(it);
            stats_.expired++;
        }
    }

    stats_.total_entries = cache_map_.size();

    if (!to_remove.empty()) {
        std::cerr << "SemanticCache: Removed " << to_remove.size() << " expired entries" << std::endl;
    }
}

SemanticCache::CacheStats SemanticCache::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace NPCInference
