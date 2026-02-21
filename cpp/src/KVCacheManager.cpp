#include "KVCacheManager.h"
#include <chrono>
#include <algorithm>
#include <iostream>

namespace NPCInference {

KVCacheManager::KVCacheManager(size_t max_memory_mb, size_t max_entries)
    : max_memory_bytes_(max_memory_mb * 1024 * 1024), max_entries_(max_entries) {}

KVCacheManager::~KVCacheManager() = default;

std::optional<KVCacheManager::CacheEntry> KVCacheManager::Get(const std::string& conversation_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(conversation_id);
    if (it == cache_map_.end()) {
        stats_.misses++;
        return std::nullopt;
    }
    
    stats_.hits++;
    UpdateAccessTime(conversation_id);
    
    // Deep copy for safety
    CacheEntry entry;
    entry.kv_tensors = CloneKV(it->second.first.kv_tensors);
    entry.sequence_length = it->second.first.sequence_length;
    entry.last_access_time = it->second.first.last_access_time;
    entry.memory_bytes = it->second.first.memory_bytes;
    
    return entry;
}

void KVCacheManager::Put(const std::string& conversation_id, 
                         std::vector<Ort::Value>&& kv_tensors,
                         size_t sequence_length) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t mem_usage = EstimateMemoryUsage(kv_tensors);
    
    // If updating existing, subtract old size
    auto it = cache_map_.find(conversation_id);
    if (it != cache_map_.end()) {
        current_memory_bytes_ -= it->second.first.memory_bytes;
    }
    
    // Evict if needed
    while ((lru_list_.size() >= max_entries_ || current_memory_bytes_ + mem_usage > max_memory_bytes_) && !lru_list_.empty()) {
        if (it != cache_map_.end() && lru_list_.size() == 1) break; 
        EvictLRU();
    }
    
    int64_t now = std::chrono::system_clock::now().time_since_epoch().count();
    CacheEntry entry;
    entry.kv_tensors = std::move(kv_tensors);
    entry.sequence_length = sequence_length;
    entry.last_access_time = now;
    entry.memory_bytes = mem_usage;
    
    if (it != cache_map_.end()) {
        it->second.first = std::move(entry);
        UpdateAccessTime(conversation_id);
    } else {
        lru_list_.push_front(conversation_id);
        cache_map_[conversation_id] = {std::move(entry), lru_list_.begin()};
    }
    
    current_memory_bytes_ += mem_usage;
    stats_.total_entries = cache_map_.size();
    stats_.total_memory_bytes = current_memory_bytes_;
}

void KVCacheManager::Remove(const std::string& conversation_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_map_.find(conversation_id);
    if (it != cache_map_.end()) {
        current_memory_bytes_ -= it->second.first.memory_bytes;
        lru_list_.erase(it->second.second);
        cache_map_.erase(it);
        stats_.total_entries = cache_map_.size();
        stats_.total_memory_bytes = current_memory_bytes_;
    }
}

void KVCacheManager::PutSystemKV(std::vector<Ort::Value>&& kv_tensors, size_t sequence_length) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t mem = EstimateMemoryUsage(kv_tensors);
    int64_t now = std::chrono::system_clock::now().time_since_epoch().count();
    
    CacheEntry entry;
    entry.kv_tensors = std::move(kv_tensors);
    entry.sequence_length = sequence_length;
    entry.last_access_time = now;
    entry.memory_bytes = mem;
    
    system_prompt_cache_ = std::make_unique<CacheEntry>(std::move(entry));
}

std::optional<KVCacheManager::CacheEntry> KVCacheManager::GetSystemKV() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!system_prompt_cache_) return std::nullopt;
    
    CacheEntry entry;
    entry.kv_tensors = CloneKV(system_prompt_cache_->kv_tensors);
    entry.sequence_length = system_prompt_cache_->sequence_length;
    entry.last_access_time = system_prompt_cache_->last_access_time;
    entry.memory_bytes = system_prompt_cache_->memory_bytes;
    
    return entry;
}

void KVCacheManager::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_map_.clear();
    lru_list_.clear();
    system_prompt_cache_.reset();
    current_memory_bytes_ = 0;
    stats_.total_entries = 0;
    stats_.total_memory_bytes = 0;
    stats_.hits = 0;
    stats_.misses = 0;
    stats_.evictions = 0;
}

KVCacheManager::CacheStats KVCacheManager::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void KVCacheManager::SetMaxMemory(size_t max_memory_mb) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_memory_bytes_ = max_memory_mb * 1024 * 1024;
    while (current_memory_bytes_ > max_memory_bytes_ && !lru_list_.empty()) {
        EvictLRU();
    }
}

std::vector<Ort::Value> KVCacheManager::CloneKV(const std::vector<Ort::Value>& source) {
    std::vector<Ort::Value> result;
    result.reserve(source.size());
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    for (const auto& v : source) {
        if (!v.IsTensor()) {
            result.push_back(Ort::Value(nullptr));
            continue;
        }

        auto info = v.GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        auto type = info.GetElementType();
        size_t count = info.GetElementCount();

        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::vector<float> data(count);
            const float* src_data = v.GetTensorData<float>();
            std::copy(src_data, src_data + count, data.data());
            
            result.push_back(Ort::Value::CreateTensor<float>(
                memory_info, data.data(), count, shape.data(), shape.size()
            ));
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            std::vector<int64_t> data(count);
            const int64_t* src_data = v.GetTensorData<int64_t>();
            std::copy(src_data, src_data + count, data.data());
            
            result.push_back(Ort::Value::CreateTensor<int64_t>(
                memory_info, data.data(), count, shape.data(), shape.size()
            ));
        } else {
            // Fallback: Copy as raw bytes if unknown but needed
            // For LLM KV caches, it's almost always F32 or Half (which we treat as bytes if not careful)
            result.push_back(Ort::Value(nullptr));
        }
    }
    return result; 
}

void KVCacheManager::EvictLRU() {
    if (lru_list_.empty()) return;
    std::string to_remove = lru_list_.back();
    auto it = cache_map_.find(to_remove);
    if (it != cache_map_.end()) {
        current_memory_bytes_ -= it->second.first.memory_bytes;
        cache_map_.erase(it);
    }
    lru_list_.pop_back();
    stats_.evictions++;
}

size_t KVCacheManager::EstimateMemoryUsage(const std::vector<Ort::Value>& tensors) {
    size_t total = 0;
    for (const auto& v : tensors) {
        if (v.IsTensor()) {
            total += v.GetTensorTypeAndShapeInfo().GetElementCount() * 4; 
        }
    }
    return total;
}

void KVCacheManager::UpdateAccessTime(const std::string& conversation_id) {
    auto it = cache_map_.find(conversation_id);
    if (it != cache_map_.end()) {
        lru_list_.erase(it->second.second);
        lru_list_.push_front(conversation_id);
        it->second.second = lru_list_.begin();
        it->second.first.last_access_time = std::chrono::system_clock::now().time_since_epoch().count();
    }
}

} // namespace NPCInference
