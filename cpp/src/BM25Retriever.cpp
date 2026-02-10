#include "BM25Retriever.h"
#include <sstream>
#include <cctype>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace NPCInference {

BM25Retriever::BM25Retriever(double k1, double b)
    : k1_(k1), b_(b) {
}

std::vector<std::string> BM25Retriever::Tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current_token;
    
    for (char c : text) {
        if (std::isalnum(c)) {
            current_token += std::tolower(c);
        } else if (!current_token.empty()) {
            tokens.push_back(current_token);
            current_token.clear();
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

void BM25Retriever::AddDocument(const std::string& doc_id, const std::string& text) {
    // Check if document already exists
    if (doc_id_to_idx_.find(doc_id) != doc_id_to_idx_.end()) {
        std::cerr << "BM25: Document " << doc_id << " already exists, skipping" << std::endl;
        return;
    }

    Document doc;
    doc.id = doc_id;
    doc.text = text;
    doc.tokens = Tokenize(text);

    // Calculate term frequencies
    for (const auto& token : doc.tokens) {
        doc.term_freq[token]++;
    }

    size_t doc_idx = documents_.size();
    doc_id_to_idx_[doc_id] = doc_idx;
    documents_.push_back(doc);

    // Update inverted index and document frequency
    for (const auto& [term, freq] : doc.term_freq) {
        inverted_index_[term].push_back({doc_idx, freq});
        
        // Update doc frequency (count unique documents containing term)
        if (freq > 0) {
            doc_freq_[term]++;
        }
    }

    // Update average document length
    double total_length = avg_doc_length_ * (documents_.size() - 1) + doc.tokens.size();
    avg_doc_length_ = total_length / documents_.size();
}

double BM25Retriever::CalculateIDF(const std::string& term) {
    auto it = doc_freq_.find(term);
    if (it == doc_freq_.end()) {
        return 0.0;
    }

    int df = it->second;
    int N = documents_.size();
    
    // IDF = log((N - df + 0.5) / (df + 0.5) + 1)
    return std::log((N - df + 0.5) / (df + 0.5) + 1.0);
}

double BM25Retriever::CalculateBM25Score(const Document& doc,
                                         const std::vector<std::string>& query_tokens) {
    double score = 0.0;
    double doc_length = doc.tokens.size();

    for (const auto& term : query_tokens) {
        auto it = doc.term_freq.find(term);
        if (it == doc.term_freq.end()) {
            continue; // Term not in document
        }

        int tf = it->second;
        double idf = CalculateIDF(term);

        // BM25 formula
        double numerator = tf * (k1_ + 1.0);
        double denominator = tf + k1_ * (1.0 - b_ + b_ * (doc_length / avg_doc_length_));
        
        score += idf * (numerator / denominator);
    }

    return score;
}

std::vector<BM25Retriever::SearchResult> BM25Retriever::Search(const std::string& query, int top_k) {
    if (documents_.empty()) {
        return {};
    }

    std::vector<std::string> query_tokens = Tokenize(query);
    if (query_tokens.empty()) {
        return {};
    }

    // Calculate scores for all documents
    std::vector<SearchResult> results;
    results.reserve(documents_.size());

    for (const auto& doc : documents_) {
        double score = CalculateBM25Score(doc, query_tokens);
        if (score > 0.0) {
            results.push_back({doc.id, score, doc.text});
        }
    }

    // Sort by score descending
    std::sort(results.begin(), results.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.score > b.score;
              });

    // Return top-k
    if (results.size() > static_cast<size_t>(top_k)) {
        results.resize(top_k);
    }

    return results;
}

void BM25Retriever::Clear() {
    documents_.clear();
    doc_id_to_idx_.clear();
    inverted_index_.clear();
    doc_freq_.clear();
    avg_doc_length_ = 0.0;
}

bool BM25Retriever::SaveIndex(const std::string& filepath) {
    try {
        json j;
        j["k1"] = k1_;
        j["b"] = b_;
        j["avg_doc_length"] = avg_doc_length_;
        
        json docs = json::array();
        for (const auto& doc : documents_) {
            json doc_json;
            doc_json["id"] = doc.id;
            doc_json["text"] = doc.text;
            docs.push_back(doc_json);
        }
        j["documents"] = docs;

        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "BM25: Failed to open file for writing: " << filepath << std::endl;
            return false;
        }

        file << j.dump(2);
        file.close();
        return true;

    } catch (const std::exception& e) {
        std::cerr << "BM25: Error saving index: " << e.what() << std::endl;
        return false;
    }
}

bool BM25Retriever::LoadIndex(const std::string& filepath) {
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "BM25: Failed to open file for reading: " << filepath << std::endl;
            return false;
        }

        json j;
        file >> j;
        file.close();

        Clear();

        k1_ = j["k1"];
        b_ = j["b"];

        // Rebuild index from documents
        for (const auto& doc_json : j["documents"]) {
            AddDocument(doc_json["id"], doc_json["text"]);
        }

        std::cerr << "BM25: Loaded " << documents_.size() << " documents" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "BM25: Error loading index: " << e.what() << std::endl;
        return false;
    }
}

} // namespace NPCInference
