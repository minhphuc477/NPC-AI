#include "HybridRetriever.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace NPCInference {

HybridRetriever::HybridRetriever(std::shared_ptr<VectorStore> vector_store,
                                 std::shared_ptr<BM25Retriever> bm25_retriever,
                                 std::shared_ptr<EmbeddingModel> embedding_model)
    : vector_store_(vector_store)
    , bm25_retriever_(bm25_retriever)
    , embedding_model_(embedding_model) {
}

void HybridRetriever::AddDocument(const std::string& doc_id, const std::string& text) {
    // Store text mapping
    doc_texts_[doc_id] = text;

    // Add to BM25 index
    bm25_retriever_->AddDocument(doc_id, text);

    // Add to vector store
    if (embedding_model_ && vector_store_) {
        std::vector<float> embedding = embedding_model_->Embed(text);
        vector_store_->Add(doc_id, embedding);
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
        std::string doc_id = std::to_string(result.id);  // Convert uint64_t to string
        
        RetrievalResult r;
        r.doc_id = doc_id;
        r.text = doc_texts_.count(doc_id) ? doc_texts_[doc_id] : "";
        r.dense_score = 1.0 - result.distance;  // Convert distance to similarity score
        r.sparse_score = 0.0;
        r.fused_score = r.dense_score;
        r.dense_rank = rank++;
        r.sparse_rank = 0;
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
    }

    // Process sparse results
    for (const auto& result : sparse_results) {
        if (doc_map.find(result.doc_id) != doc_map.end()) {
            // Document in both results
            doc_map[result.doc_id].sparse_score = result.sparse_score;
            doc_map[result.doc_id].sparse_rank = result.sparse_rank;
            doc_map[result.doc_id].fused_score += 
                config.sparse_weight * CalculateRRFScore(result.sparse_rank, config.rrf_k);
        } else {
            // Document only in sparse results
            doc_map[result.doc_id] = result;
            doc_map[result.doc_id].fused_score = 
                config.sparse_weight * CalculateRRFScore(result.sparse_rank, config.rrf_k);
        }
    }

    // Convert map to vector
    std::vector<RetrievalResult> fused_results;
    fused_results.reserve(doc_map.size());
    for (const auto& [doc_id, result] : doc_map) {
        if (result.fused_score >= config.min_score_threshold) {
            fused_results.push_back(result);
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

void HybridRetriever::Clear() {
    doc_texts_.clear();
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

    return success;
}

} // namespace NPCInference
