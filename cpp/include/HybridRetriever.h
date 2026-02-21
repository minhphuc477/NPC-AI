#pragma once

#include "VectorStore.h"
#include "BM25Retriever.h"
#include "EmbeddingModel.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace NPCInference {

/**
 * Hybrid Retriever combining dense and sparse search
 * 
 * Features:
 * - Dense vector search (semantic similarity)
 * - Sparse BM25 search (keyword matching)
 * - Reciprocal Rank Fusion (RRF) for result merging
 * - Adaptive relevance thresholds
 */
class HybridRetriever {
public:
    struct RetrievalResult {
        std::string doc_id;
        std::string text;
        double dense_score;
        double sparse_score;
        double fused_score;
        int dense_rank;
        int sparse_rank;
    };

    struct RetrievalConfig {
        int top_k = 5;
        double dense_weight = 0.6;
        double sparse_weight = 0.4;
        double min_score_threshold = 0.0;
        int rrf_k = 60;
    };

    /**
     * Constructor
     */
    HybridRetriever(std::shared_ptr<VectorStore> vector_store,
                    std::shared_ptr<BM25Retriever> bm25_retriever,
                    std::shared_ptr<EmbeddingModel> embedding_model);

    /**
     * Add document to both dense and sparse indices
     */
    void AddDocument(const std::string& doc_id, const std::string& text);

    /**
     * Hybrid search with RRF fusion
     */
    std::vector<RetrievalResult> Search(const std::string& query,
                                        const RetrievalConfig& config);
    
    // Overload for default config
    std::vector<RetrievalResult> Search(const std::string& query);

    /**
     * Dense-only search
     */
    std::vector<RetrievalResult> DenseSearch(const std::string& query, int top_k = 5);

    /**
     * Sparse-only search
     */
    std::vector<RetrievalResult> SparseSearch(const std::string& query, int top_k = 5);

    /**
     * Clear all indices
     */
    void Clear();

    /**
     * Get document count
     */
    size_t GetDocumentCount() const;

    /**
     * Save indices
     */
    bool SaveIndices(const std::string& base_path);

    /**
     * Load indices
     */
    bool LoadIndices(const std::string& base_path);

private:
    // Reciprocal Rank Fusion
    std::vector<RetrievalResult> FuseResults(
        const std::vector<RetrievalResult>& dense_results,
        const std::vector<RetrievalResult>& sparse_results,
        const RetrievalConfig& config);

    // Calculate RRF score
    double CalculateRRFScore(int rank, int k);

    std::shared_ptr<VectorStore> vector_store_;
    std::shared_ptr<BM25Retriever> bm25_retriever_;
    std::shared_ptr<EmbeddingModel> embedding_model_;

    // Document ID to text mapping
    std::unordered_map<std::string, std::string> doc_texts_;
};

} // namespace NPCInference
