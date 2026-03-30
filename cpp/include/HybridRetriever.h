#pragma once

#include "VectorStore.h"
#include "BM25Retriever.h"
#include "EmbeddingModel.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <map>

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
        double base_fused_score = 0.0;
        double trust_score = 1.0;
        double injection_risk = 0.0;
        int dense_rank;
        int sparse_rank;
    };

    struct RetrievalConfig {
        int top_k = 5;
        double dense_weight = 0.6;
        double sparse_weight = 0.4;
        double min_score_threshold = 0.0;
        int rrf_k = 60;

        // Robustness guard against retrieval poisoning / prompt injection.
        bool enable_robustness_guard = true;
        double trust_weight = 0.35;
        double min_trust_score = 0.15;
        double injection_penalty_scale = 0.6;
        double max_injection_risk = 0.95;

        // Novelty component: state-conditioned guard strength lambda(s).
        bool state_conditioned_guard_enabled = true;
        std::string guard_behavior_state;
        double guard_lambda_conflict = 1.25;
        double guard_lambda_task = 1.0;
        double guard_lambda_social = 0.80;

        // Novelty component: state-conditioned dense/sparse fusion alpha(s).
        bool state_conditioned_fusion_enabled = true;
        double dense_weight_conflict = 0.72;
        double dense_weight_task = 0.58;
        double dense_weight_social = 0.65;

        // Novelty component: query-aware lexical adaptation alpha(s, q).
        // For low lexical overlap (paraphrase-heavy queries), boost dense branch.
        // For high lexical overlap (keyword-heavy queries), boost sparse branch.
        bool query_aware_fusion_enabled = true;
        double lexical_low_overlap_threshold = 0.22;
        double lexical_high_overlap_threshold = 0.55;
        double lexical_dense_boost_low_overlap = 0.12;
        double lexical_sparse_boost_high_overlap = 0.12;
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
    void AddDocument(const std::string& doc_id,
                     const std::string& text,
                     const std::map<std::string, std::string>& metadata = {});

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
        const RetrievalConfig& config,
        double effective_dense_weight,
        double effective_sparse_weight);

    // Calculate RRF score
    double CalculateRRFScore(int rank, int k);
    double ComputeTrustScore(const std::map<std::string, std::string>& metadata) const;
    double ComputeInjectionRisk(
        const std::string& text,
        const std::map<std::string, std::string>& metadata
    ) const;
    double LexicalOverlapRatio(const std::string& query, const std::string& text) const;
    std::pair<double, double> ApplyQueryAwareFusion(
        const std::string& query,
        const std::vector<RetrievalResult>& sparse_results,
        const RetrievalConfig& config,
        double dense_weight,
        double sparse_weight
    ) const;
    double GuardLambdaForState(const RetrievalConfig& config) const;
    std::pair<double, double> ResolveFusionWeights(const RetrievalConfig& config) const;
    double Clamp01(double value) const;

    std::shared_ptr<VectorStore> vector_store_;
    std::shared_ptr<BM25Retriever> bm25_retriever_;
    std::shared_ptr<EmbeddingModel> embedding_model_;

    // Document ID to text mapping
    std::unordered_map<std::string, std::string> doc_texts_;
    std::unordered_map<std::string, std::map<std::string, std::string>> doc_metadata_;
};

} // namespace NPCInference
