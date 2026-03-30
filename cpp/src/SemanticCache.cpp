#include "SemanticCache.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#if NPC_USE_USEARCH
#include <usearch/index_dense.hpp>
#endif

namespace NPCInference {

struct SemanticCache::Impl {
#if NPC_USE_USEARCH
    unum::usearch::index_dense_gt<uint64_t, uint32_t> idx;
#endif
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

#if NPC_USE_USEARCH
    // Use usearch's SIMD distance calculation when available.
    unum::usearch::metric_punned_t metric(a.size(), unum::usearch::metric_kind_t::cos_k, unum::usearch::scalar_kind_t::f32_k);
    float distance = metric((const unum::usearch::byte_t*)a.data(), (const unum::usearch::byte_t*)b.data());
    
    // Convert distance back to similarity: similarity = 1 - distance.
    return 1.0f - distance;
#else
    // Portable fallback.
    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na <= 1e-8f || nb <= 1e-8f) {
        return 0.0f;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
#endif
}

std::optional<SemanticCache::CacheEntry> SemanticCache::Get(const std::string& query) {
    if (!embedding_model_) {
        return std::nullopt;
    }

    // Generate query embedding WITHOUT LOCK
    std::vector<float> query_embedding = embedding_model_->Embed(query);
    if (query_embedding.empty()) {
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_map_.empty()) {
        stats_.misses++;
        return std::nullopt;
    }

    int64_t current_time = GetCurrentTimestamp();

#if NPC_USE_USEARCH
    // If index is unavailable, this is a miss.
    if (!impl_->idx) {
        stats_.misses++;
        return std::nullopt;
    }

    // Search in index for Top 5 candidates.
    auto matches = impl_->idx.search(query_embedding.data(), 5);
    if (matches.size() == 0) {
        stats_.misses++;
        return std::nullopt;
    }

    // Iterate through candidates to find the first valid match and GC expired ones
    for (std::size_t i = 0; i < matches.size(); ++i) {
        uint64_t candidate_id = matches[i].member.key;
        float candidate_distance = matches[i].distance;
        float candidate_similarity = 1.0f - candidate_distance;

        // Skip immediately if below threshold
        if (candidate_similarity < similarity_threshold_) continue;

        auto id_it = id_to_query_.find(candidate_id);
        if (id_it == id_to_query_.end()) continue;

        std::string candidate_query = id_it->second;
        auto cache_it = cache_map_.find(candidate_query);
        if (cache_it == cache_map_.end()) continue;

        auto& entryNode = cache_it->second;

        // Check if expired
        if (entryNode.entry.ttl_seconds > 0 && 
            (current_time - entryNode.entry.timestamp) > entryNode.entry.ttl_seconds) {
            // Lazy Garbage Collection: Object has expired. Remove it immediately.
            lru_list_.erase(entryNode.lru_it);
            if (impl_->idx) impl_->idx.remove(candidate_id);
            id_to_query_.erase(candidate_id);
            cache_map_.erase(cache_it);
            stats_.expired++;
            stats_.total_entries = cache_map_.size();
            continue; // Continue searching next candidates
        }

        // We found the best valid match!
        stats_.hits++;

        // Update LRU
        lru_list_.erase(entryNode.lru_it);
        lru_list_.push_front(entryNode.usearch_id);
        entryNode.lru_it = lru_list_.begin();
        entryNode.entry.hit_count++;

        return entryNode.entry;
    }
#else
    // Brute-force fallback without usearch.
    std::string best_query;
    float best_similarity = -std::numeric_limits<float>::infinity();

    for (auto it = cache_map_.begin(); it != cache_map_.end();) {
        auto& entry_node = it->second;
        const auto& entry = entry_node.entry;
        bool expired = entry.ttl_seconds > 0 && (current_time - entry.timestamp) > entry.ttl_seconds;
        if (expired) {
            lru_list_.erase(entry_node.lru_it);
            id_to_query_.erase(entry_node.usearch_id);
            it = cache_map_.erase(it);
            stats_.expired++;
            stats_.total_entries = cache_map_.size();
            continue;
        }

        const float sim = CosineSimilarity(query_embedding, entry.query_embedding);
        if (sim >= similarity_threshold_ && sim > best_similarity) {
            best_similarity = sim;
            best_query = it->first;
        }
        ++it;
    }

    if (!best_query.empty()) {
        auto cache_it = cache_map_.find(best_query);
        if (cache_it != cache_map_.end()) {
            stats_.hits++;
            lru_list_.erase(cache_it->second.lru_it);
            lru_list_.push_front(cache_it->second.usearch_id);
            cache_it->second.lru_it = lru_list_.begin();
            cache_it->second.entry.hit_count++;
            return cache_it->second.entry;
        }
    }
#endif

    stats_.misses++;
    return std::nullopt;
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

    // Initialize usearch if needed.
#if NPC_USE_USEARCH
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
#endif

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
        lru_list_.push_front(it->second.usearch_id);
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

    // Add to index if usearch is enabled.
#if NPC_USE_USEARCH
    if (impl_->idx) {
        impl_->idx.add(new_id, node.entry.query_embedding.data());
    }
#endif

    // Add to LRU front
    lru_list_.push_front(new_id);
    node.lru_it = lru_list_.begin();

    cache_map_[query] = std::move(node);
    id_to_query_[new_id] = query;
    stats_.total_entries = cache_map_.size();
}

void SemanticCache::EvictLRU() {
    if (lru_list_.empty()) return;

    // Remove least recently used (back of list)
    uint64_t vid = lru_list_.back();
    lru_list_.pop_back();

    auto qt = id_to_query_.find(vid);
    if (qt != id_to_query_.end()) {
        std::string victim = qt->second;
        auto it = cache_map_.find(victim);
        if (it != cache_map_.end()) {
#if NPC_USE_USEARCH
            if (impl_->idx) impl_->idx.remove(vid);
#endif
            id_to_query_.erase(vid);
            cache_map_.erase(it);
        }
    }

    stats_.evictions++;
    stats_.total_entries = cache_map_.size();
}

void SemanticCache::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_map_.clear();
    id_to_query_.clear();
    lru_list_.clear();
#if NPC_USE_USEARCH
    if (impl_->idx) {
        impl_->idx.clear();
    }
#endif
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
#if NPC_USE_USEARCH
            if (impl_->idx) impl_->idx.remove(vid);
#endif
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
