#include "VectorStore.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <shared_mutex>

// Production Implementation utilizing USearch
// Only compiled when NPC_USE_MOCKS=OFF
#include <usearch/index_dense.hpp>

using json = nlohmann::json;
using namespace unum::usearch;

namespace NPCInference {

    struct VectorStore::Impl {
        // usearch index for cosine similarity, f32
        index_dense_gt<uint64_t, uint32_t> idx;
        std::shared_mutex rw_mutex;
        
        Impl() {}
    };

    VectorStore::VectorStore() : impl_(std::make_unique<Impl>()) {}
    VectorStore::~VectorStore() = default;

    bool VectorStore::Initialize(size_t dimension) {
        try {
            // Configure metric
            metric_punned_t metric(dimension, metric_kind_t::cos_k, scalar_kind_t::f32_k);
            
            // Configure index
            index_dense_config_t config;
            config.connectivity = 16;
            config.expansion_add = 128;
            config.expansion_search = 64;

            impl_->idx = index_dense_gt<uint64_t, uint32_t>::make(metric, config);
            
            if (!impl_->idx) {
                std::cerr << "VectorStore: Failed to create index." << std::endl;
                return false;
            }

            // Reserve initial capacity to avoid crashes during Add
            impl_->idx.reserve(10000);
            
            std::cout << "VectorStore: Initialized (Dim=" << dimension << ")." << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "VectorStore Init Error: " << e.what() << std::endl;
            return false;
        }
    }

    void VectorStore::Add(const std::string& text, const std::vector<float>& embedding, const std::map<std::string, std::string>& metadata) {
        std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
        uint64_t id = next_id_++;
        documents_[id] = {text, metadata};
        
        // Add to index
        if (impl_->idx) {
            auto result = impl_->idx.add(id, embedding.data());
            if (!result) {
                std::cerr << "VectorStore: Failed to add document " << id << ": " << result.error.what() << std::endl;
            }
        }
    }

    std::vector<SearchResult> VectorStore::Search(const std::vector<float>& query, size_t k) {
        std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
        std::vector<SearchResult> results;
        if (documents_.empty() || !impl_->idx) return results;
        
        // Search
        auto result = impl_->idx.search(query.data(), k);
        
        // Iterate results
        for (std::size_t i = 0; i != result.size(); ++i) {
            auto match = result[i];
            uint64_t id = match.member.key;
            float distance = match.distance;
            
            // Cosine distance range [0, 2]. Similarity = 1 - distance
            // usearch cos: result is distance (1 - cos).
            float similarity = 1.0f - distance;

            if (documents_.count(id)) {
                results.push_back({
                    id,
                    documents_[id].text,
                    similarity,
                    documents_[id].metadata
                });
            }
        }
        return results;
    }

    bool VectorStore::Save(const std::string& path_prefix) {
        std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
        try {
            if (!impl_->idx) return false;
            
            // Save Index
            std::string idx_path = path_prefix + ".usearch";
            auto res = impl_->idx.save(unum::usearch::output_file_t(idx_path.c_str()), unum::usearch::index_dense_serialization_config_t{}, unum::usearch::dummy_progress_t{});
            if (!res) {
                std::cerr << "VectorStore: Failed to save index: " << res.error.what() << std::endl;
                return false;
            }

            // Save Metadata (Binary MessagePack)
            std::string doc_path = path_prefix + ".msgpack";
            json j;
            j["version"] = 2; // version 2 uses msgpack
            json docs_json;
            for (const auto& [id, doc] : documents_) {
                docs_json[std::to_string(id)] = {
                    {"text", doc.text},
                    {"meta", doc.metadata}
                };
            }
            j["docs"] = docs_json;
            j["next_id"] = next_id_.load();
            
            std::vector<uint8_t> msgpack_data = json::to_msgpack(j);
            std::ofstream f(doc_path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(msgpack_data.data()), msgpack_data.size());
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "VectorStore Save Error: " << e.what() << std::endl;
            return false;
        }
    }

    bool VectorStore::Load(const std::string& path_prefix) {
        std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
        try {
            // Load Index
            std::string idx_path = path_prefix + ".usearch";
            if (std::filesystem::exists(idx_path)) {
                 auto res = impl_->idx.load(unum::usearch::input_file_t(idx_path.c_str()), unum::usearch::index_dense_serialization_config_t{}, unum::usearch::dummy_progress_t{});
                 if (!res) {
                     std::cerr << "VectorStore: Failed to load index: " << res.error.what() << std::endl;
                 }
            }

            // Load Metadata (try msgpack first, fallback to JSON)
            std::string msgpack_path = path_prefix + ".msgpack";
            std::string json_path = path_prefix + ".json";
            json j;
            
            if (std::filesystem::exists(msgpack_path)) {
                std::ifstream f(msgpack_path, std::ios::binary);
                std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                j = json::from_msgpack(buffer);
            } else if (std::filesystem::exists(json_path)) {
                std::ifstream f(json_path);
                j = json::parse(f);
            } else {
                return true; // No metadata, clean state
            }
            
            if (j.contains("docs")) {
                for (const auto& el : j["docs"].items()) {
                    uint64_t id = std::stoull(el.key());
                    std::string text = el.value()["text"];
                    std::map<std::string, std::string> meta;
                    if (el.value().contains("meta")) {
                         meta = el.value()["meta"].get<std::map<std::string, std::string>>();
                    }
                    documents_[id] = {text, meta};
                }
            }
            if (j.contains("next_id")) {
                next_id_.store(j["next_id"]);
            }
            return true;
        } catch (const std::exception& e) {
             std::cerr << "VectorStore Load Error: " << e.what() << std::endl;
             return false;
        }
    }

    std::vector<SearchResult> VectorStore::GetAllMemories() {
        std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
        std::vector<SearchResult> all_docs;
         for (const auto& [id, doc] : documents_) {
            all_docs.push_back({id, doc.text, 1.0f, doc.metadata});
        }
        return all_docs;
    }

    void VectorStore::Remove(uint64_t id) {
        std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
        documents_.erase(id);
        if (impl_->idx) {
            impl_->idx.remove(id);
        }
    }
}
