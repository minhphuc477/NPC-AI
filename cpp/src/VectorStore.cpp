#include "VectorStore.h"
#include "HalfFloat.h"

// Define macros before including usearch
#define USEARCH_USE_FP16 0
#define USEARCH_USE_FP16LIB 0

// Workaround for missing _Float16 support on MSVC
// We define it as our HalfFloat struct
#if defined(_MSC_VER) && !defined(__clang__)
#define _Float16 NPCInference::HalfFloat
#endif
#include <usearch/index_dense.hpp>
#include <fstream>
#include <iostream>

using namespace unum::usearch;

namespace NPCInference {

    struct VectorStore::Impl {
        // Use index_dense_gt to handle metrics and serialization automatically
        index_dense_gt<> index;
        
        Impl() : index() {}
    };

    VectorStore::VectorStore() : impl_(std::make_unique<Impl>()) {}
    VectorStore::~VectorStore() = default;

    bool VectorStore::Initialize(size_t dimension) {
        try {
            // metric_kind_t is defined in index_plugins.hpp which index_dense.hpp includes
            // we use index_dense_gt<>::metric_t for type safety
            using metric_t = index_dense_gt<>::metric_t;
            
            metric_t metric(dimension, metric_kind_t::cos_k);
            impl_->index = index_dense_gt<>::make(metric);
            
            // reserve not strictly needed for dense index but good practice if available
            // index_dense_gt might not expose reserve directly?
            // it has reserve(limits).
            // impl_->index.reserve(1000); 
            return true;
        } catch (const std::exception& e) {
            std::cerr << "VectorStore Init Error: " << e.what() << std::endl;
            return false;
        }
    }

    void VectorStore::Add(const std::string& text, const std::vector<float>& embedding, const std::map<std::string, std::string>& metadata) {
        if (embedding.empty()) return;

        uint64_t id = next_id_++;
        
        try {
            // index_dense_gt add takes (key, vector)
            // vector is float* for f32 metric
            impl_->index.add(id, embedding.data());
            
            // Store metadata
            documents_[id] = {text, metadata};
        } catch (const std::exception& e) {
             std::cerr << "VectorStore Add Error: " << e.what() << std::endl;
        }
    }

    std::vector<SearchResult> VectorStore::Search(const std::vector<float>& query, size_t k) {
        std::vector<SearchResult> results;
        if (query.empty() || documents_.empty()) return results;

        try {
            auto search_results = impl_->index.search(query.data(), k);
            
            for (std::size_t i = 0; i < search_results.size(); ++i) {
                // search_results[i] returns a match_t which has member.key (if using index_dense_gt?)
                // Wait, index_dense_gt::search returns search_result_t
                // search_result_t iteration returns match object with key and distance
                auto match = search_results[i];
                uint64_t id = match.member.key; 
                float distance = match.distance;
                
                if (documents_.count(id)) {
                    const auto& doc = documents_.at(id);
                    results.push_back({id, doc.text, distance, doc.metadata});
                }
            }
        } catch (const std::exception& e) {
             std::cerr << "VectorStore Search Error: " << e.what() << std::endl;
        }
        
        return results;
    }

    bool VectorStore::Save(const std::string& path_prefix) {
        try {
            // 1. Save usearch index
            // index_dense_gt SHOULD have save(path)
            // If not, we will see compile error and fix it.
            // Based on other methods, it likely has.
            // If not, we can use save_to_stream.
            std::string index_path = path_prefix + ".usearch";
            impl_->index.save(index_path.c_str());
            
            // 2. Save metadata (JSON)
            std::string meta_path = path_prefix + ".json";
            nlohmann::json j;
            j["next_id"] = next_id_;
            j["docs"] = nlohmann::json::object();
            
            for (const auto& [id, doc] : documents_) {
                j["docs"][std::to_string(id)] = {
                    {"text", doc.text},
                    {"meta", doc.metadata}
                };
            }
            
            std::ofstream f(meta_path);
            f << j.dump(4);
            
            return true;
        } catch (const std::exception& e) {
             std::cerr << "VectorStore Save Error: " << e.what() << std::endl;
             return false;
        }
    }

    bool VectorStore::Load(const std::string& path_prefix) {
        try {
            // 1. Load usearch index
            std::string index_path = path_prefix + ".usearch";
            std::string meta_path = path_prefix + ".json";
            
            // For loading, we might need to know dimension if not in file?
            // index_dense_gt load checks header.
            // We can use static make(path) to load?
            // Or impl_->index.load(path).
            
            // If impl_->index is empty/uninit, load() should work if file has header.
            impl_->index.load(index_path.c_str());
            
            // 2. Load metadata
            std::ifstream f(meta_path);
            if (!f.is_open()) return false;
            nlohmann::json j;
            f >> j;
            
            next_id_ = j["next_id"];
            documents_.clear();
            
            for (const auto& el : j["docs"].items()) {
                uint64_t id = std::stoull(el.key());
                std::string text = el.value()["text"];
                std::map<std::string, std::string> meta = el.value()["meta"];
                documents_[id] = {text, meta};
            }
            
            return true;
        } catch (const std::exception& e) {
             std::cerr << "VectorStore Load Error: " << e.what() << std::endl;
             return false;
        }
    }

} // namespace NPCInference
