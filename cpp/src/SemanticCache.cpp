#include "SemanticCache.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <usearch/index_dense.hpp>

namespace NPCInference {

struct SemanticCache::Impl {
    unum::usearch::index_dense_gt<uint64_t, uint32_t> idx;
};

SemanticCache::SemanticCache(std::shared_ptr<EmbeddingModel> embedding_model,
                             float similarity_threshold,
                             size_t max_entries,
                             int64_t default_ttl_seconds)
    : embedding_model_(embedding_model)
    , similarity_threshold_(similarity_threshold)
    , max_entries_(max_entries)
    , default_ttl_seconds_(default_ttl_seconds)
    , impl_(std::make_unique<Impl>()) {
}

SemanticCache::~SemanticCache() = default;

int64_t SemanticCache::GetCurrentTimestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

float SemanticCache::CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0f;
    }

    // Use usearch's highly optimized SIMD distance calculation
    unum::usearch::metric_punned_t metric(a.size(), unum::usearch::metric_kind_t::cos_k, unum::usearch::scalar_kind_t::f32_k);
    float distance = metric((const unum::usearch::byte_t*)a.data(), (const unum::usearch::byte_t*)b.data());
    
    // Convert distance back to similarity: similarity = 1 - distance
    return 1.0f - distance;
}

const SemanticCache::CacheEntry* SemanticCache::Get(const std::string& query) {
    if (!embedding_model_) {
        return nullptr;
    }

    // Generate query embedding WITHOUT LOCK
    std::vector<float> query_embedding = embedding_model_->Embed(query);
    if (query_embedding.empty()) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // If cache is empty or index not initialized, miss
    if (cache_map_.empty() || !impl_->idx) {
        stats_.misses++;
        // Do not spam stdout std::cerr << "SemanticCache: MISS (query='" << query << "')" << std::endl;
        return nullptr;
    }

    // Search in index
    auto matches = impl_->idx.search(query_embedding.data(), 1);
    if (matches.size() == 0) {
        stats_.misses++;
        return nullptr;
    }
    
    uint64_t best_id = matches[0].member.key;
    float best_distance = matches[0].distance;
    float best_similarity = 1.0f - best_distance; // usearch cos_k distance is essentially 1 - cos_sim
    
    // Fallback manual similarity calculation if we need exactly cosine (usearch matches closely)
    // Actually best_similarity is perfectly correlated
    
    std::string best_query;
    auto id_it = id_to_query_.find(best_id);
    if (id_it != id_to_query_.end()) {
        best_query = id_it->second;
    } else {
        stats_.misses++;
        return nullptr;
    }

    auto cache_it = cache_map_.find(best_query);
    if (cache_it == cache_map_.end()) {
        stats_.misses++;
        return nullptr;
    }

    auto& entryNode = cache_it->second;

    int64_t current_time = GetCurrentTimestamp();
    // Check if expired
    if (entryNode.entry.ttl_seconds > 0 && 
        (current_time - entryNode.entry.timestamp) > entryNode.entry.ttl_seconds) {
        // Lazy Garland Collection: Object has expired. Remove it immediately.
        lru_list_.erase(entryNode.lru_it);
        if (impl_->idx) impl_->idx.remove(best_id);
        id_to_query_.erase(best_id);
        cache_map_.erase(cache_it);
        stats_.expired++;
        stats_.total_entries = cache_map_.size();
        
        stats_.misses++;
        return nullptr;
    }

    // Check if best match exceeds threshold
    if (best_similarity >= similarity_threshold_) {
        stats_.hits++;

        // Update LRU
        lru_list_.erase(entryNode.lru_it);
        lru_list_.push_front(best_query);
        entryNode.lru_it = lru_list_.begin();
        entryNode.entry.hit_count++;

        // Optional log: std::cerr << "SemanticCache: HIT (similarity=" << best_similarity << ")\n";

        return &(entryNode.entry);
    }

    stats_.misses++;
    return nullptr;
}

void SemanticCache::Put(const std::string& query, const std::string& result, int64_t ttl_seconds) {
    if (!embedding_model_) {
        return;
    }

    // Generate query embedding WITHOUT LOCK
    std::vector<float> query_embedding = embedding_model_->Embed(query);
    if (query_embedding.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Initialize usearch if needed
    if (!impl_->idx) {
        unum::usearch::metric_punned_t metric(query_embedding.size(), unum::usearch::metric_kind_t::cos_k, unum::usearch::scalar_kind_t::f32_k);
        unum::usearch::index_dense_config_t config;
        config.connectivity = 16;
        config.expansion_add = 64;
        config.expansion_search = 32;
        impl_->idx = unum::usearch::index_dense_gt<uint64_t, uint32_t>::make(metric, config);
        if (impl_->idx) {
            impl_->idx.reserve(max_entries_);
        } else {
            std::cerr << "SemanticCache: Error creating usearch index" << std::endl;
            return;
        }
    }

    // Use default TTL if not specified
    if (ttl_seconds < 0) {
        ttl_seconds = default_ttl_seconds_;
    }

    // Check if query already exists
    auto it = cache_map_.find(query);
    if (it != cache_map_.end()) {
        // Update existing entry
        it->second.entry.result = result;
        it->second.entry.timestamp = GetCurrentTimestamp();
        it->second.entry.ttl_seconds = ttl_seconds;

        // Move to front of LRU
        lru_list_.erase(it->second.lru_it);
        lru_list_.push_front(query);
        it->second.lru_it = lru_list_.begin();

        return;
    }

    // Evict if at capacity
    while (cache_map_.size() >= max_entries_) {
        EvictLRU();
    }

    // Create new entry
    uint64_t new_id = next_id_++;
    
    CacheNode node;
    node.entry.query = query;
    node.entry.query_embedding = std::move(query_embedding);
    node.entry.result = result;
    node.entry.timestamp = GetCurrentTimestamp();
    node.entry.ttl_seconds = ttl_seconds;
    node.entry.hit_count = 0;
    node.usearch_id = new_id;

    // Add to index
    if (impl_->idx) {
        impl_->idx.add(new_id, node.entry.query_embedding.data());
    }

    // Add to LRU front
    lru_list_.push_front(query);
    node.lru_it = lru_list_.begin();

    cache_map_[query] = std::move(node);
    id_to_query_[new_id] = query;
    stats_.total_entries = cache_map_.size();
}

void SemanticCache::EvictLRU() {
    if (lru_list_.empty()) return;

    // Remove least recently used (back of list)
    std::string victim = lru_list_.back();
    lru_list_.pop_back();

    auto it = cache_map_.find(victim);
    if (it != cache_map_.end()) {
        uint64_t vid = it->second.usearch_id;
        if (impl_->idx) impl_->idx.remove(vid);
        id_to_query_.erase(vid);
        cache_map_.erase(it);
    }

    stats_.evictions++;
    stats_.total_entries = cache_map_.size();
}

void SemanticCache::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_map_.clear();
    id_to_query_.clear();
    lru_list_.clear();
    if (impl_->idx) {
        impl_->idx.clear();
    }
    stats_.total_entries = 0;
}

void SemanticCache::RemoveExpired() {
    std::lock_guard<std::mutex> lock(mutex_);

    int64_t current_time = GetCurrentTimestamp();
    std::vector<std::string> to_remove;

    for (const auto& [query, node] : cache_map_) {
        const auto& entry = node.entry;
        if (entry.ttl_seconds > 0 && 
            (current_time - entry.timestamp) > entry.ttl_seconds) {
            to_remove.push_back(query);
        }
    }

    for (const auto& query : to_remove) {
        auto it = cache_map_.find(query);
        if (it != cache_map_.end()) {
            lru_list_.erase(it->second.lru_it);
            uint64_t vid = it->second.usearch_id;
            if (impl_->idx) impl_->idx.remove(vid);
            id_to_query_.erase(vid);
            cache_map_.erase(it);
            stats_.expired++;
        }
    }

    stats_.total_entries = cache_map_.size();
}

SemanticCache::CacheStats SemanticCache::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace NPCInference
