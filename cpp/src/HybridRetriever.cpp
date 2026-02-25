#ifdef _WIN32
#include <windows.h>
#undef GetCurrentTime
#endif

#include "HybridRetriever.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace NPCInference {

namespace {

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool Contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

bool ContainsAny(const std::string& haystack, const std::initializer_list<const char*>& needles) {
    for (const char* needle : needles) {
        if (Contains(haystack, needle)) {
            return true;
        }
    }
    return false;
}

} // namespace

HybridRetriever::HybridRetriever(std::shared_ptr<VectorStore> vector_store,
                                 std::shared_ptr<BM25Retriever> bm25_retriever,
                                 std::shared_ptr<EmbeddingModel> embedding_model)
    : vector_store_(vector_store)
    , bm25_retriever_(bm25_retriever)
    , embedding_model_(embedding_model) {
}

void HybridRetriever::AddDocument(const std::string& doc_id,
                                  const std::string& text,
                                  const std::map<std::string, std::string>& metadata) {
    // Store text mapping
    doc_texts_[doc_id] = text;
    std::map<std::string, std::string> merged_metadata = metadata;
    merged_metadata["doc_id"] = doc_id;
    if (!merged_metadata.count("source")) {
        merged_metadata["source"] = "memory";
    }
    doc_metadata_[doc_id] = merged_metadata;

    // Add to BM25 index
    bm25_retriever_->AddDocument(doc_id, text);

    // Add to vector store
    if (embedding_model_ && vector_store_) {
        std::vector<float> embedding = embedding_model_->Embed(text);
        std::cerr << "HybridRetriever: Adding to VectorStore (" << text.length() << " chars, " << embedding.size() << " dims)" << std::endl;
        vector_store_->Add(text, embedding, merged_metadata);
    }
}

std::vector<HybridRetriever::RetrievalResult> HybridRetriever::DenseSearch(
    const std::string& query, int top_k) {
    
    std::vector<RetrievalResult> results;

    if (!embedding_model_ || !vector_store_) {
        return results;
    }

    // Generate query embedding
    std::vector<float> query_embedding = embedding_model_->Embed(query);

    // Search vector store
    auto search_results = vector_store_->Search(query_embedding, top_k);

    // Convert to RetrievalResult
    int rank = 1;
    for (const auto& result : search_results) {
        std::string doc_id;
        if (result.metadata.count("doc_id")) {
            doc_id = result.metadata.at("doc_id");
        } else {
            doc_id = std::to_string(result.id);
        }
        
        RetrievalResult r;
        r.doc_id = doc_id;
        r.text = result.text;
        r.dense_score = 1.0 - result.distance;  // Convert distance to similarity score
        r.sparse_score = 0.0;
        r.fused_score = r.dense_score;
        r.base_fused_score = r.fused_score;
        std::map<std::string, std::string> merged_metadata = result.metadata;
        r.trust_score = ComputeTrustScore(merged_metadata);
        r.dense_rank = rank++;
        r.sparse_rank = 0;

        auto meta_it = doc_metadata_.find(doc_id);
        if (meta_it != doc_metadata_.end()) {
            for (const auto& [key, value] : meta_it->second) {
                merged_metadata[key] = value;
            }
            r.trust_score = std::min(r.trust_score, ComputeTrustScore(meta_it->second));
        }
        r.injection_risk = ComputeInjectionRisk(result.text, merged_metadata);

        results.push_back(r);
    }

    return results;
}

std::vector<HybridRetriever::RetrievalResult> HybridRetriever::SparseSearch(
    const std::string& query, int top_k) {
    
    std::vector<RetrievalResult> results;

    if (!bm25_retriever_) {
        return results;
    }

    // BM25 search
    auto bm25_results = bm25_retriever_->Search(query, top_k);

    // Convert to RetrievalResult
    int rank = 1;
    for (const auto& result : bm25_results) {
        RetrievalResult r;
        r.doc_id = result.doc_id;
        r.text = result.text;
        r.dense_score = 0.0;
        r.sparse_score = result.score;
        r.fused_score = result.score;
        r.base_fused_score = r.fused_score;
        std::map<std::string, std::string> metadata;
        auto meta_it = doc_metadata_.find(result.doc_id);
        if (meta_it != doc_metadata_.end()) {
            metadata = meta_it->second;
            r.trust_score = ComputeTrustScore(meta_it->second);
        } else {
            r.trust_score = 0.75; // Unknown sparse-only provenance.
        }
        r.injection_risk = ComputeInjectionRisk(result.text, metadata);
        r.dense_rank = 0;
        r.sparse_rank = rank++;
        results.push_back(r);
    }

    return results;
}

double HybridRetriever::CalculateRRFScore(int rank, int k) {
    if (rank == 0) return 0.0;
    return 1.0 / (k + rank);
}

std::vector<HybridRetriever::RetrievalResult> HybridRetriever::FuseResults(
    const std::vector<RetrievalResult>& dense_results,
    const std::vector<RetrievalResult>& sparse_results,
    const RetrievalConfig& config) {
    
    // Collect all unique documents
    std::unordered_map<std::string, RetrievalResult> doc_map;

    // Process dense results
    for (const auto& result : dense_results) {
        doc_map[result.doc_id] = result;
        doc_map[result.doc_id].fused_score = 
            config.dense_weight * CalculateRRFScore(result.dense_rank, config.rrf_k);
        doc_map[result.doc_id].base_fused_score = doc_map[result.doc_id].fused_score;
    }

    // Process sparse results
    for (const auto& result : sparse_results) {
        if (doc_map.find(result.doc_id) != doc_map.end()) {
            // Document in both results
            doc_map[result.doc_id].sparse_score = result.sparse_score;
            doc_map[result.doc_id].sparse_rank = result.sparse_rank;
            doc_map[result.doc_id].fused_score += 
                config.sparse_weight * CalculateRRFScore(result.sparse_rank, config.rrf_k);
            doc_map[result.doc_id].base_fused_score = doc_map[result.doc_id].fused_score;
            doc_map[result.doc_id].trust_score = std::min(doc_map[result.doc_id].trust_score, result.trust_score);
            doc_map[result.doc_id].injection_risk = std::max(doc_map[result.doc_id].injection_risk, result.injection_risk);
        } else {
            // Document only in sparse results
            doc_map[result.doc_id] = result;
            doc_map[result.doc_id].fused_score = 
                config.sparse_weight * CalculateRRFScore(result.sparse_rank, config.rrf_k);
            doc_map[result.doc_id].base_fused_score = doc_map[result.doc_id].fused_score;
        }
    }

    // Convert map to vector
    std::vector<RetrievalResult> fused_results;
    fused_results.reserve(doc_map.size());
    int dropped_by_guard = 0;
    const double trust_weight = Clamp01(config.trust_weight);
    const double risk_scale = Clamp01(config.injection_penalty_scale);

    for (const auto& [doc_id, result] : doc_map) {
        RetrievalResult adjusted = result;

        if (config.enable_robustness_guard) {
            if (adjusted.trust_score < config.min_trust_score ||
                adjusted.injection_risk > config.max_injection_risk) {
                dropped_by_guard++;
                continue;
            }

            const double trust_factor = (1.0 - trust_weight) + (trust_weight * Clamp01(adjusted.trust_score));
            const double risk_penalty = std::max(0.0, 1.0 - (risk_scale * Clamp01(adjusted.injection_risk)));
            adjusted.fused_score = adjusted.base_fused_score * trust_factor * risk_penalty;
        }

        if (adjusted.fused_score >= config.min_score_threshold) {
            fused_results.push_back(adjusted);
        }
    }

    // Sort by fused score descending
    std::sort(fused_results.begin(), fused_results.end(),
              [](const RetrievalResult& a, const RetrievalResult& b) {
                  return a.fused_score > b.fused_score;
              });

    // Return top-k
    if (fused_results.size() > static_cast<size_t>(config.top_k)) {
        fused_results.resize(config.top_k);
    }

    if (config.enable_robustness_guard && dropped_by_guard > 0) {
        std::cerr << "HybridRetriever: robustness guard dropped " << dropped_by_guard
                  << " candidate docs (trust/risk filter)." << std::endl;
    }

    return fused_results;
}

std::vector<HybridRetriever::RetrievalResult> HybridRetriever::Search(
    const std::string& query, const RetrievalConfig& config) {
    
    // Perform both searches
    auto dense_results = DenseSearch(query, config.top_k * 2);  // Fetch more for fusion
    auto sparse_results = SparseSearch(query, config.top_k * 2);

    // Fuse results using RRF
    auto fused_results = FuseResults(dense_results, sparse_results, config);

    std::cerr << "HybridRetriever: Query='" << query << "', "
              << "Dense=" << dense_results.size() << ", "
              << "Sparse=" << sparse_results.size() << ", "
              << "Fused=" << fused_results.size() << std::endl;

    return fused_results;
}

std::vector<HybridRetriever::RetrievalResult> HybridRetriever::Search(const std::string& query) {
    return Search(query, RetrievalConfig());
}

void HybridRetriever::Clear() {
    doc_texts_.clear();
    doc_metadata_.clear();
    // Note: VectorStore doesn't have Clear() method
    if (bm25_retriever_) bm25_retriever_->Clear();
}

size_t HybridRetriever::GetDocumentCount() const {
    return doc_texts_.size();
}

bool HybridRetriever::SaveIndices(const std::string& base_path) {
    bool success = true;

    // Save BM25 index
    if (bm25_retriever_) {
        success &= bm25_retriever_->SaveIndex(base_path + "_bm25.json");
    }

    // Save vector store
    if (vector_store_) {
        success &= vector_store_->Save(base_path + "_vectors.bin");
    }

    return success;
}

bool HybridRetriever::LoadIndices(const std::string& base_path) {
    bool success = true;

    // Load BM25 index
    if (bm25_retriever_) {
        success &= bm25_retriever_->LoadIndex(base_path + "_bm25.json");
    }

    // Load vector store
    if (vector_store_) {
        success &= vector_store_->Load(base_path + "_vectors.bin");
    }

    // Rebuild doc_texts_ from BM25 (since it has the text)
    // This is a simplification - in production, save/load doc_texts_ separately
    doc_texts_.clear();
    doc_metadata_.clear();

    return success;
}

double HybridRetriever::Clamp01(double value) const {
    return std::max(0.0, std::min(1.0, value));
}

double HybridRetriever::ComputeTrustScore(const std::map<std::string, std::string>& metadata) const {
    auto explicit_trust = metadata.find("trust_score");
    if (explicit_trust != metadata.end()) {
        try {
            return Clamp01(std::stod(explicit_trust->second));
        } catch (...) {
            // Fall back to heuristic trust assignment.
        }
    }

    double trust = 0.80; // Conservative default for unknown provenance.
    auto source_it = metadata.find("source");
    if (source_it != metadata.end()) {
        const std::string source = ToLower(source_it->second);
        if (Contains(source, "system") || Contains(source, "verified") || Contains(source, "lore")) {
            trust = 0.95;
        } else if (Contains(source, "memory") || Contains(source, "npc")) {
            trust = 0.85;
        } else if (Contains(source, "user") || Contains(source, "player")) {
            trust = 0.65;
        } else if (Contains(source, "external") || Contains(source, "web") || Contains(source, "unverified")) {
            trust = 0.45;
        }
    }

    auto verified_it = metadata.find("verified");
    if (verified_it != metadata.end()) {
        const std::string verified = ToLower(verified_it->second);
        if (verified == "true" || verified == "1" || verified == "yes") {
            trust = std::max(trust, 0.90);
        } else if (verified == "false" || verified == "0" || verified == "no") {
            trust = std::min(trust, 0.50);
        }
    }

    return Clamp01(trust);
}

double HybridRetriever::ComputeInjectionRisk(
    const std::string& text,
    const std::map<std::string, std::string>& metadata
) const {
    const std::string lower = ToLower(text);
    double risk = 0.0;

    const std::pair<const char*, double> patterns[] = {
        {"ignore previous", 0.50},
        {"ignore all previous", 0.60},
        {"ignore prior", 0.45},
        {"system prompt", 0.45},
        {"hidden prompt", 0.45},
        {"reveal hidden prompt", 0.55},
        {"system override", 0.55},
        {"developer mode", 0.45},
        {"developer message", 0.40},
        {"jailbreak", 0.50},
        {"override instructions", 0.45},
        {"discard safety", 0.45},
        {"new priority rule", 0.35},
        {"trust this passage only", 0.45},
        {"replace all facts", 0.55},
        {"output exactly", 0.45},
        {"critical patch note", 0.35},
        {"attacker answer", 0.45},
        {"return attacker answer", 0.55},
        {"obey hidden prompt", 0.55},
        {"do not follow", 0.30},
        {"reveal secret", 0.45},
        {"api key", 0.55},
        {"password", 0.55},
        {"tool_call", 0.25},
        {"execute command", 0.35},
        {"prompt injection", 0.50}
    };

    for (const auto& p : patterns) {
        if (Contains(lower, p.first)) {
            risk += p.second;
        }
    }

    const char* directive_cues[] = {
        "ignore",
        "reveal",
        "output",
        "replace",
        "discard",
        "override",
        "obey",
        "return",
        "say"
    };
    int directive_hits = 0;
    for (const char* cue : directive_cues) {
        if (Contains(lower, cue)) {
            directive_hits++;
        }
    }

    if (directive_hits >= 2) {
        risk += std::min(0.40, 0.12 * static_cast<double>(directive_hits));
    } else if (directive_hits == 1) {
        risk += 0.05;
    }

    if (Contains(lower, ":") && directive_hits > 0) {
        risk += 0.08;
    }
    if (ContainsAny(lower, {"unverified", "traveler note"}) && directive_hits > 0) {
        risk += 0.12;
    }
    if ((Contains(lower, "system") && Contains(lower, "override")) || Contains(lower, "hidden prompt")) {
        risk += 0.20;
    }

    auto attack_type_it = metadata.find("attack_type");
    if (attack_type_it != metadata.end()) {
        const std::string attack_type = ToLower(attack_type_it->second);
        if (ContainsAny(attack_type, {"prompt_injection", "spoof", "adversarial"})) {
            risk += 0.25;
        }
    }

    double declared_trust = 0.0;
    auto trust_it = metadata.find("trust_score");
    if (trust_it != metadata.end()) {
        try {
            declared_trust = Clamp01(std::stod(trust_it->second));
        } catch (...) {
            declared_trust = 0.0;
        }
    }

    auto source_it = metadata.find("source");
    if (source_it != metadata.end()) {
        const std::string source = ToLower(source_it->second);
        const bool claims_verified = ContainsAny(source, {"verified", "system", "lore"});
        if (claims_verified && directive_hits > 0 && risk >= 0.30) {
            // Metadata/text inconsistency: a "verified" document should not carry imperative override instructions.
            risk += 0.15;
        }
    }

    if (declared_trust >= 0.85 && directive_hits > 0 && risk >= 0.35) {
        // Spoof-resistance: high-trust claims do not neutralize strong injection cues.
        risk += 0.20;
    }

    if (Contains(lower, "```") &&
        (Contains(lower, "system") || Contains(lower, "assistant") || directive_hits > 0)) {
        risk += 0.15;
    }

    return Clamp01(risk);
}

} // namespace NPCInference
