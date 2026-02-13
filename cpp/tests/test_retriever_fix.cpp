#include "HybridRetriever.h"
#include "VectorStore.h"
#include "BM25Retriever.h"
#include "EmbeddingModel.h"
#include <iostream>
#include <cassert>
#include <filesystem>

// Mock Embedding Model
class MockEmbeddingModel : public NPCInference::EmbeddingModel {
public:
    virtual bool Load(const std::string&, const std::string&) override { return true; }
    virtual bool IsLoaded() const override { return true; }
    virtual std::vector<float> Embed(const std::string& text) override {
        return std::vector<float>(384, 0.1f); 
    }
};

// Mock Vector Store
class MockVectorStore : public NPCInference::VectorStore {
public:
    struct Doc {
        uint64_t id;
        std::string text;
        std::map<std::string, std::string> metadata;
    };
    std::vector<Doc> docs;
    
    virtual bool Initialize(size_t) override { return true; }
    
    virtual void Add(const std::string& text, const std::vector<float>&, const std::map<std::string, std::string>& metadata) override {
        std::cout << "MockVS: Adding " << text << std::endl;
        docs.push_back({(uint64_t)docs.size() + 1, text, metadata}); // ID starts at 1
    }
    
    virtual std::vector<NPCInference::SearchResult> Search(const std::vector<float>&, size_t) override {
        std::vector<NPCInference::SearchResult> results;
        for (const auto& doc : docs) {
            results.push_back({doc.id, doc.text, 0.1f, doc.metadata});
        }
        return results;
    }
    
    virtual bool Save(const std::string&) override { return true; }
    virtual bool Load(const std::string&) override { return true; }
};

void TestHybridPersistence() {
    // Note: We are testing HybridRetriever LOGIC, not VectorStore persistence (which is mocked).
    // Specifically, we test that HybridRetriever passes correct text/metadata to VectorStore
    // and correctly retrieves it back.
    
    std::string test_dir = "test_data_mock";
    std::filesystem::create_directory(test_dir);
    
    auto vs = std::make_shared<MockVectorStore>();
    
    auto bm25 = std::make_shared<NPCInference::BM25Retriever>();
    auto embed = std::make_shared<MockEmbeddingModel>();
    
    NPCInference::HybridRetriever retriever(vs, bm25, embed);
    
    // 1. Add Document
    std::cout << "Adding document..." << std::endl;
    retriever.AddDocument("doc1", "This is a test document about AI.");
    
    // Verify VectorStore received it correctly
    assert(vs->docs.size() == 1);
    assert(vs->docs[0].text == "This is a test document about AI.");
    assert(vs->docs[0].metadata.count("doc_id"));
    assert(vs->docs[0].metadata["doc_id"] == "doc1");
    std::cout << "VectorStore received correct data." << std::endl;
    
    // 2. Search
    std::cout << "Searching..." << std::endl;
    auto results = retriever.DenseSearch("AI", 1);
    
    std::cout << "Results: " << results.size() << std::endl;
    if (!results.empty()) {
        std::cout << "Text: " << results[0].text << std::endl;
        assert(results[0].text == "This is a test document about AI.");
        std::cout << "DocID: " << results[0].doc_id << std::endl;
        assert(results[0].doc_id == "doc1");
        std::cout << "SUCCESS: HybridRetriever logic verified." << std::endl;
    } else {
        std::cout << "FAILURE: No results found." << std::endl;
        assert(false);
    }
    
    // Cleanup
    std::filesystem::remove_all(test_dir);
}

int main() {
    try {
        TestHybridPersistence();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
