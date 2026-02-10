#include "KVCacheManager.h"
#include <chrono>
#include <iostream>
#include <algorithm>

namespace NPCInference {

KVCacheManager::KVCacheManager(size_t max_memory_mb, size_t max_entries)
    : max_memory_bytes_(max_memory_mb * 1024 * 1024)
    , max_entries_(max_entries) {
    std::cerr << "KVCacheManager initialized: max_memory=" << max_memory_mb 
              << "MB, max_entries=" << max_entries << std::endl;
}

KVCacheManager::~KVCacheManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cerr << "KVCacheManager stats - Hits: " << stats_.hits 
              << ", Misses: " << stats_.misses 
              << ", Hit Rate: " << (stats_.hit_rate() * 100.0f) << "%"
              << ", Evictions: " << stats_.evictions << std::endl;
}

KVCacheManager::CacheEntry* KVCacheManager::Get(const std::string& conversation_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_map_.find(conversation_id);
    if (it == cache_map_.end()) {
        stats_.misses++;
        return nullptr;
    }
    
    stats_.hits++;
    UpdateAccessTime(conversation_id);
    return &(it->second.first);
}

void KVCacheManager::Put(const std::string& conversation_id,
                         std::vector<Ort::Value>&& kv_tensors,
                         size_t sequence_length) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t memory_needed = EstimateMemoryUsage(kv_tensors);
    
    // Evict if necessary
    while ((current_memory_bytes_ + memory_needed > max_memory_bytes_ || 
            cache_map_.size() >= max_entries_) && !cache_map_.empty()) {
        EvictLRU();
    }
    
    // Remove old entry if updating
    auto it = cache_map_.find(conversation_id);
    if (it != cache_map_.end()) {
        current_memory_bytes_ -= it->second.first.memory_bytes;
        lru_list_.erase(it->second.second);
        cache_map_.erase(it);
    }
    
    // Create new entry
    CacheEntry entry;
    entry.kv_tensors = std::move(kv_tensors);
    entry.sequence_length = sequence_length;
    entry.last_access_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    entry.memory_bytes = memory_needed;
    
    // Add to LRU front (most recent)
    lru_list_.push_front(conversation_id);
    auto lru_it = lru_list_.begin();
    
    cache_map_[conversation_id] = {std::move(entry), lru_it};
    current_memory_bytes_ += memory_needed;
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

void KVCacheManager::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_map_.clear();
    lru_list_.clear();
    current_memory_bytes_ = 0;
    stats_.total_entries = 0;
    stats_.total_memory_bytes = 0;
}

KVCacheManager::CacheStats KVCacheManager::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void KVCacheManager::SetMaxMemory(size_t max_memory_mb) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_memory_bytes_ = max_memory_mb * 1024 * 1024;
    
    // Evict if over new limit
    while (current_memory_bytes_ > max_memory_bytes_ && !cache_map_.empty()) {
        EvictLRU();
    }
}

void KVCacheManager::EvictLRU() {
    if (lru_list_.empty()) return;
    
    // Remove least recently used (back of list)
    std::string victim = lru_list_.back();
    lru_list_.pop_back();
    
    auto it = cache_map_.find(victim);
    if (it != cache_map_.end()) {
        current_memory_bytes_ -= it->second.first.memory_bytes;
        cache_map_.erase(it);
        stats_.evictions++;
        stats_.total_entries = cache_map_.size();
        stats_.total_memory_bytes = current_memory_bytes_;
        
        std::cerr << "KVCache: Evicted conversation " << victim 
                  << " (LRU), current memory: " << (current_memory_bytes_ / 1024 / 1024) << "MB" << std::endl;
    }
}

size_t KVCacheManager::EstimateMemoryUsage(const std::vector<Ort::Value>& tensors) {
    size_t total_bytes = 0;
    
    for (const auto& tensor : tensors) {
        if (tensor.IsTensor()) {
            auto shape_info = tensor.GetTensorTypeAndShapeInfo();
            auto shape = shape_info.GetShape();
            
            // Calculate element count
            size_t element_count = 1;
            for (auto dim : shape) {
                element_count *= static_cast<size_t>(dim);
            }
            
            // Estimate bytes (assuming float32 = 4 bytes)
            auto elem_type = shape_info.GetElementType();
            size_t bytes_per_element = 4; // Default to float32
            
            switch (elem_type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                    bytes_per_element = 2;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    bytes_per_element = 4;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                    bytes_per_element = 8;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                    bytes_per_element = 1;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                    bytes_per_element = 2;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
                    bytes_per_element = 4;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
                    bytes_per_element = 8;
                    break;
                default:
                    break;
            }
            
            total_bytes += element_count * bytes_per_element;
        }
    }
    
    return total_bytes;
}

void KVCacheManager::UpdateAccessTime(const std::string& conversation_id) {
    auto it = cache_map_.find(conversation_id);
    if (it == cache_map_.end()) return;
    
    // Move to front of LRU list
    lru_list_.erase(it->second.second);
    lru_list_.push_front(conversation_id);
    it->second.second = lru_list_.begin();
    
    // Update timestamp
    it->second.first.last_access_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

} // namespace NPCInference
