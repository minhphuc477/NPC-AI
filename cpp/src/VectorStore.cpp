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
        
        DocData newDoc;
        newDoc.text = text;
        newDoc.metadata = metadata;
        
        // AAA Production: Initialize decay metadata
        auto now = std::chrono::system_clock::now();
        newDoc.timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
        newDoc.importance = metadata.count("importance") ? std::stof(metadata.at("importance")) : 1.0f;
        
        documents_[id] = newDoc;
        
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
                    {"meta", doc.metadata},
                    {"timestamp", doc.timestamp},
                    {"importance", doc.importance}
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

    std::future<bool> VectorStore::SaveAsync(const std::string& path_prefix) {
        // Return a future that runs the existing Save function asynchronously.
        // Save uses std::shared_lock, so it won't block Search operations.
        // It will only temporarily block Add operations until the background I/O completes.
        return std::async(std::launch::async, [this, path_prefix]() {
            return this->Save(path_prefix);
        });
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

            // SAX Parser to prevent json::parse DOM memory blowout
            class VectorStoreSAX : public json::json_sax_t {
            public:
                std::unordered_map<uint64_t, DocData>& docs;
                uint64_t& next_id;
                
                int depth = 0;
                std::string current_key;
                uint64_t current_doc_id = 0;
                std::string current_text;
                std::map<std::string, std::string> current_meta;
                std::string current_meta_key;
                unsigned long long current_timestamp = 0;
                float current_importance = 1.0f;
                bool in_docs = false;
                bool in_doc_meta = false;

                VectorStoreSAX(std::unordered_map<uint64_t, DocData>& d, uint64_t& nid) : docs(d), next_id(nid) {}

                bool null() override { return true; }
                bool boolean(bool val) override { return true; }
                bool number_integer(number_integer_t val) override { 
                    if (depth == 1 && current_key == "next_id") next_id = val; 
                    else if (in_docs && depth == 3 && current_key == "timestamp") current_timestamp = val;
                    return true; 
                }
                bool number_unsigned(number_unsigned_t val) override { 
                    if (depth == 1 && current_key == "next_id") next_id = val; 
                    else if (in_docs && depth == 3 && current_key == "timestamp") current_timestamp = val;
                    return true; 
                }
                bool number_float(number_float_t val, const string_t& s) override { 
                    if (in_docs && depth == 3 && current_key == "importance") current_importance = val;
                    return true; 
                }
                bool string(string_t& val) override { 
                    if (in_docs) {
                        if (depth == 3 && current_key == "text") current_text = val;
                        else if (in_doc_meta && depth == 4) current_meta[current_meta_key] = val;
                    }
                    return true; 
                }
                bool binary(json::binary_t& val) override { return true; }
                bool start_object(std::size_t elements) override { depth++; return true; }
                bool key(string_t& val) override { 
                    if (depth == 1) { current_key = val; if (val == "docs") in_docs = true; }
                    else if (depth == 2 && in_docs) { 
                        current_doc_id = std::stoull(val); 
                        current_text.clear(); current_meta.clear(); 
                        current_timestamp = 0; current_importance = 1.0f;
                    }
                    else if (depth == 3 && in_docs) { current_key = val; if (val == "meta") in_doc_meta = true; }
                    else if (depth == 4 && in_doc_meta) { current_meta_key = val; }
                    return true; 
                }
                bool end_object() override { 
                    if (depth == 4 && in_doc_meta) in_doc_meta = false;
                    else if (depth == 3 && in_docs) docs[current_doc_id] = {current_text, current_meta, current_timestamp, current_importance}; // Commit doc
                    else if (depth == 2 && in_docs) in_docs = false;
                    depth--; return true; 
                }
                bool start_array(std::size_t elements) override { return true; }
                bool end_array() override { return true; }
                bool parse_error(std::size_t pos, const std::string& token, const json::exception& ex) override { return false; }
            };

            uint64_t loaded_next_id = next_id_.load();
            VectorStoreSAX sax_handler(documents_, loaded_next_id);
            
            std::string msgpack_path = path_prefix + ".msgpack";
            std::string json_path = path_prefix + ".json";
            
            if (std::filesystem::exists(msgpack_path)) {
                std::ifstream f(msgpack_path, std::ios::binary);
                std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                json::sax_parse(buffer, &sax_handler, json::input_format_t::msgpack);
            } else if (std::filesystem::exists(json_path)) {
                std::ifstream f(json_path);
                json::sax_parse(f, &sax_handler);
            } else {
                return true; // No metadata, clean state
            }
            
            next_id_.store(loaded_next_id);
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

    int VectorStore::Prune(float decay_rate, float min_importance) {
        std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
        int pruned_count = 0;
        
        std::vector<uint64_t> to_remove;
        
        for (auto& [id, doc] : documents_) {
            // Memory Decay: exponential decay of importance
            doc.importance *= decay_rate;
            
            // Delete if faded
            if (doc.importance < min_importance) {
                to_remove.push_back(id);
            }
        }
        
        for (uint64_t id : to_remove) {
            documents_.erase(id);
            if (impl_->idx) impl_->idx.remove(id);
            pruned_count++;
        }
        
        if (pruned_count > 0) {
            std::cout << "Memory Decay: Pruned " << pruned_count << " faded memories." << std::endl;
        }
        
        return pruned_count;
    }
}
