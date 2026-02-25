#pragma once

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <future>
#include <nlohmann/json.hpp>

// Forward declare usearch index type to avoid pulling heavy headers into public interface if possible
// But usearch is header-only templates, so we usually include it.
// To keep compile times low, we can use PIMPL or just include it.
// Given it's a template, we might need to include it or wrap it in cpp.
// Let's use PIMPL idiom to hide usearch details.

namespace NPCInference {

    struct SearchResult {
        uint64_t id;
        std::string text;
        float distance;
        std::map<std::string, std::string> metadata;
    };

    class VectorStore {
    public:
        VectorStore();
        virtual ~VectorStore();

        // Initialize with dimension (e.g. 384 for MiniLM)
        virtual bool Initialize(size_t dimension);

        // Add document
        virtual void Add(const std::string& text, const std::vector<float>& embedding, const std::map<std::string, std::string>& metadata = {});

        // Search
        virtual std::vector<SearchResult> Search(const std::vector<float>& query, size_t k);

        // Persistence
        virtual bool Save(const std::string& path_prefix);
        virtual bool Load(const std::string& path_prefix);

        // Management (Phase 10)
        virtual std::vector<SearchResult> GetAllMemories();
        virtual void Remove(uint64_t id);

        // AAA Phase 5: Memory Decay & Pruning
        // @param decay_rate multiplier to simulate time passing (e.g. importance *= decay_rate)
        // @param min_importance threshold below which memory is deleted
        virtual int Prune(float decay_rate = 0.95f, float min_importance = 0.1f);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        
        // Metadata storage (ID -> Data)
        // We use a simple atomic counter for IDs
        std::atomic<uint64_t> next_id_{1};
        
        struct DocData {
            std::string text;
            std::map<std::string, std::string> metadata;
            std::vector<float> embedding;
            
            // For Memory Decay
            unsigned long long timestamp = 0;
            float importance = 1.0f; 
        };
        
        std::unordered_map<uint64_t, DocData> documents_;
    };

} // namespace NPCInference
