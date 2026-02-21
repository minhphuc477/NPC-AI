#include "VectorStore.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

// Production Implementation utilizing USearch
// Only compiled when NPC_USE_MOCKS=OFF
#include <usearch/index_dense.hpp>

using json = nlohmann::json;
using namespace unum::usearch;

namespace NPCInference {

    struct VectorStore::Impl {
        // usearch index for cosine similarity, f32
        index_dense_gt<uint64_t, uint32_t> idx;
        
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
        try {
            if (!impl_->idx) return false;
            
            // Save Index
            std::string idx_path = path_prefix + ".usearch";
            auto res = impl_->idx.save(unum::usearch::output_file_t(idx_path.c_str()), unum::usearch::index_dense_serialization_config_t{}, unum::usearch::dummy_progress_t{});
            if (!res) {
                std::cerr << "VectorStore: Failed to save index: " << res.error.what() << std::endl;
                return false;
            }

            // Save Metadata (JSON)
            std::string doc_path = path_prefix + ".json";
            json j;
            j["version"] = 1;
            json docs_json;
            for (const auto& [id, doc] : documents_) {
                docs_json[std::to_string(id)] = {
                    {"text", doc.text},
                    {"meta", doc.metadata}
                };
            }
            j["docs"] = docs_json;
            j["next_id"] = next_id_;
            
            std::ofstream f(doc_path);
            f << j.dump(2);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "VectorStore Save Error: " << e.what() << std::endl;
            return false;
        }
    }

    bool VectorStore::Load(const std::string& path_prefix) {
        try {
            // Load Index
            std::string idx_path = path_prefix + ".usearch";
            if (std::filesystem::exists(idx_path)) {
                 auto res = impl_->idx.load(unum::usearch::input_file_t(idx_path.c_str()), unum::usearch::index_dense_serialization_config_t{}, unum::usearch::dummy_progress_t{});
                 if (!res) {
                     std::cerr << "VectorStore: Failed to load index: " << res.error.what() << std::endl;
                 }
            }

            // Load Metadata
            std::string doc_path = path_prefix + ".json";
            if (!std::filesystem::exists(doc_path)) return true;
            
            std::ifstream f(doc_path);
            json j = json::parse(f);
            
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
                next_id_ = j["next_id"];
            }
            return true;
        } catch (const std::exception& e) {
             std::cerr << "VectorStore Load Error: " << e.what() << std::endl;
             return false;
        }
    }

    std::vector<SearchResult> VectorStore::GetAllMemories() {
        std::vector<SearchResult> all_docs;
         for (const auto& [id, doc] : documents_) {
            all_docs.push_back({id, doc.text, 1.0f, doc.metadata});
        }
        return all_docs;
    }

    void VectorStore::Remove(uint64_t id) {
        documents_.erase(id);
        if (impl_->idx) {
            impl_->idx.remove(id);
        }
    }
}
