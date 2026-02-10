#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cmath>
#include <algorithm>

namespace NPCInference {

/**
 * BM25 Sparse Retriever for keyword-based search
 * 
 * Implements BM25 (Best Matching 25) algorithm for sparse retrieval.
 * Complements dense vector search with keyword matching.
 */
class BM25Retriever {
public:
    struct Document {
        std::string id;
        std::string text;
        std::vector<std::string> tokens;
        std::unordered_map<std::string, int> term_freq;
    };

    struct SearchResult {
        std::string doc_id;
        double score;
        std::string text;
    };

    /**
     * Constructor
     * @param k1 Term frequency saturation parameter (default: 1.5)
     * @param b Document length normalization parameter (default: 0.75)
     */
    explicit BM25Retriever(double k1 = 1.5, double b = 0.75);

    /**
     * Add document to index
     */
    void AddDocument(const std::string& doc_id, const std::string& text);

    /**
     * Search for top-k documents
     */
    std::vector<SearchResult> Search(const std::string& query, int top_k = 5);

    /**
     * Clear all documents
     */
    void Clear();

    /**
     * Get document count
     */
    size_t GetDocumentCount() const { return documents_.size(); }

    /**
     * Save index to file
     */
    bool SaveIndex(const std::string& filepath);

    /**
     * Load index from file
     */
    bool LoadIndex(const std::string& filepath);

private:
    // Tokenize text into words
    std::vector<std::string> Tokenize(const std::string& text);

    // Calculate IDF for a term
    double CalculateIDF(const std::string& term);

    // Calculate BM25 score for a document
    double CalculateBM25Score(const Document& doc, 
                              const std::vector<std::string>& query_tokens);

    // BM25 parameters
    double k1_;
    double b_;

    // Document storage
    std::vector<Document> documents_;
    std::unordered_map<std::string, size_t> doc_id_to_idx_;

    // Inverted index: term -> list of (doc_idx, term_freq)
    std::unordered_map<std::string, std::vector<std::pair<size_t, int>>> inverted_index_;

    // Document frequency: term -> number of documents containing term
    std::unordered_map<std::string, int> doc_freq_;

    // Average document length
    double avg_doc_length_ = 0.0;
};

} // namespace NPCInference
