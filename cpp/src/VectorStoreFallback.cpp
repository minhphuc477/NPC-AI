#include "VectorStore.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <shared_mutex>

using json = nlohmann::json;

namespace NPCInference {

struct VectorStore::Impl {
    size_t dimension = 0;
    bool initialized = false;
    mutable std::shared_mutex rw_mutex;
};

namespace {

float Dot(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

float Norm(const std::vector<float>& v) {
    float s = 0.0f;
    for (float x : v) s += x * x;
    return std::sqrt(std::max(0.0f, s));
}

float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    const float na = Norm(a);
    const float nb = Norm(b);
    if (na <= 1e-8f || nb <= 1e-8f) return 0.0f;
    return Dot(a, b) / (na * nb);
}

} // namespace

VectorStore::VectorStore() : impl_(std::make_unique<Impl>()) {}
VectorStore::~VectorStore() = default;

bool VectorStore::Initialize(size_t dimension) {
    std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
    impl_->dimension = dimension;
    impl_->initialized = true;
    std::cout << "VectorStore (fallback): Initialized (Dim=" << dimension << ")." << std::endl;
    return true;
}

void VectorStore::Add(
    const std::string& text,
    const std::vector<float>& embedding,
    const std::map<std::string, std::string>& metadata
) {
    std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
    const uint64_t id = next_id_++;

    DocData doc;
    doc.text = text;
    doc.metadata = metadata;
    doc.embedding = embedding;

    auto now = std::chrono::system_clock::now();
    doc.timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    doc.importance = metadata.count("importance") ? std::stof(metadata.at("importance")) : 1.0f;

    documents_[id] = std::move(doc);
}

std::vector<SearchResult> VectorStore::Search(const std::vector<float>& query, size_t k) {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    std::vector<SearchResult> out;
    if (documents_.empty() || query.empty()) return out;

    out.reserve(documents_.size());
    for (const auto& [id, doc] : documents_) {
        if (doc.embedding.empty()) continue;
        const float sim = CosineSimilarity(query, doc.embedding);
        out.push_back({id, doc.text, sim, doc.metadata});
    }

    std::sort(
        out.begin(),
        out.end(),
        [](const SearchResult& a, const SearchResult& b) { return a.distance > b.distance; }
    );

    if (out.size() > k) out.resize(k);
    return out;
}

bool VectorStore::Save(const std::string& path_prefix) {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    try {
        json j;
        j["version"] = 3;
        j["dimension"] = impl_->dimension;
        j["next_id"] = next_id_.load();
        j["docs"] = json::object();
        for (const auto& [id, doc] : documents_) {
            j["docs"][std::to_string(id)] = {
                {"text", doc.text},
                {"meta", doc.metadata},
                {"embedding", doc.embedding},
                {"timestamp", doc.timestamp},
                {"importance", doc.importance}
            };
        }

        const std::vector<uint8_t> payload = json::to_msgpack(j);
        std::ofstream f(path_prefix + ".msgpack", std::ios::binary);
        f.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
        return true;
    } catch (const std::exception& e) {
        std::cerr << "VectorStore (fallback) Save Error: " << e.what() << std::endl;
        return false;
    }
}

bool VectorStore::Load(const std::string& path_prefix) {
    std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
    try {
        std::ifstream f(path_prefix + ".msgpack", std::ios::binary);
        if (!f.good()) return true; // clean state

        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (bytes.empty()) return true;

        const json j = json::from_msgpack(bytes);
        impl_->dimension = j.value("dimension", impl_->dimension);

        documents_.clear();
        if (j.contains("docs") && j["docs"].is_object()) {
            for (auto it = j["docs"].begin(); it != j["docs"].end(); ++it) {
                const uint64_t id = std::stoull(it.key());
                const json& d = it.value();
                DocData doc;
                doc.text = d.value("text", "");
                doc.metadata = d.value("meta", std::map<std::string, std::string>{});
                if (d.contains("embedding") && d["embedding"].is_array()) {
                    doc.embedding = d["embedding"].get<std::vector<float>>();
                }
                doc.timestamp = d.value("timestamp", 0ULL);
                doc.importance = d.value("importance", 1.0f);
                documents_[id] = std::move(doc);
            }
        }

        const uint64_t loaded_next = j.value("next_id", next_id_.load());
        next_id_.store(std::max<uint64_t>(loaded_next, static_cast<uint64_t>(documents_.size() + 1)));
        impl_->initialized = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "VectorStore (fallback) Load Error: " << e.what() << std::endl;
        return false;
    }
}

std::vector<SearchResult> VectorStore::GetAllMemories() {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    std::vector<SearchResult> all;
    all.reserve(documents_.size());
    for (const auto& [id, doc] : documents_) {
        all.push_back({id, doc.text, 1.0f, doc.metadata});
    }
    return all;
}

void VectorStore::Remove(uint64_t id) {
    std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
    documents_.erase(id);
}

int VectorStore::Prune(float decay_rate, float min_importance) {
    std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
    int pruned = 0;
    std::vector<uint64_t> to_remove;
    for (auto& [id, doc] : documents_) {
        doc.importance *= decay_rate;
        if (doc.importance < min_importance) {
            to_remove.push_back(id);
        }
    }
    for (uint64_t id : to_remove) {
        documents_.erase(id);
        ++pruned;
    }
    return pruned;
}

} // namespace NPCInference
