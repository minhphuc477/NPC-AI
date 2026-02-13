#include "NPCInference.h"
#include "ModelLoader.h"
#include "Tokenizer.h"
#include "EmbeddingModel.h"
#include "VectorStore.h"
#include "HybridRetriever.h"
#include "BM25Retriever.h"
#include <iostream>
#include <cassert>
#include <memory>
#include <algorithm>

using namespace NPCInference;

// --- Mocks ---

class MockModelLoader : public ModelLoader {
public:
    virtual bool LoadModel(const std::string&, bool, int) override { return true; }
    virtual bool IsLoaded() const override { return true; }
    
    virtual std::vector<int64_t> Generate(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>&,
        int,
        const std::string&,
        std::function<void(int64_t)> callback,
        std::function<void(float*, int64_t)> logit_processor
    ) override {
        // Simulate generation
        // If logit_processor is present (GrammarSampler), we should test it?
        // Hard to test void callback side effects without complex setup.
        // We will just return a dummy response.
        
        if (logit_processor) {
            std::cout << "MockModel: Logit processor received (GrammarSampler active)." << std::endl;
            // Simulate calling it to ensure it doesn't crash
            float logits[100];
            std::fill(logits, logits + 100, 0.0f);
            logit_processor(logits, 100);
        }
        
        std::vector<int64_t> output = input_ids;
        std::vector<int64_t> new_tokens = {1, 2, 3};
        
        if (callback) {
            for(auto id : new_tokens) callback(id);
        }
        
        output.insert(output.end(), new_tokens.begin(), new_tokens.end());
        return output;
    }
};

class MockTokenizer : public Tokenizer {
public:
    virtual bool Load(const std::string&) override { return true; }
    virtual bool IsLoaded() const override { return true; }
    virtual std::vector<int64_t> Encode(const std::string& text) override { return {10, 11, 12}; }
    virtual std::string Decode(const std::vector<int64_t>& ids) override { return "Mock Response"; }
    virtual int GetVocabSize() const override { return 100; }
    virtual int GetEOSId() const override { return 99; }
};

class MockEmbeddingModel : public EmbeddingModel {
public:
    virtual bool Load(const std::string&, const std::string&) override { return true; }
    virtual bool IsLoaded() const override { return true; }
    virtual std::vector<float> Embed(const std::string&) override { return std::vector<float>(384, 0.1f); }
};

class MockVectorStore : public VectorStore {
public:
    virtual bool Initialize(size_t) override { return true; }
    virtual bool Load(const std::string&) override { return true; }
    virtual void Add(const std::string& text, const std::vector<float>&, const std::map<std::string, std::string>&) override { 
        std::cout << "MockVectorStore: Added " << text << std::endl;
        memories_.push_back({next_id_++, text, 0.0f, {}});
    }
    virtual std::vector<SearchResult> Search(const std::vector<float>&, size_t) override { return {}; }
    
    virtual std::vector<SearchResult> GetAllMemories() override {
        return memories_;
    }
    
    virtual void Remove(uint64_t id) override {
        std::cout << "MockVectorStore: Removed " << id << std::endl;
    }
    
    // Simple in-memory storage for test
    std::vector<SearchResult> memories_;
    uint64_t next_id_ = 1;
};

// --- Test Subclass accessing protected members if needed ---
// Actually NPCInferenceEngine has private members. 
// We will test public interface basically, but we need to inject mocks.
// NPCInferenceEngine constructor hardcodes instantiation. 
// We need a way to inject mocks. 
// Since we didn't add dependency injection, we might need a "TestableNPCInferenceEngine" 
// that bypasses standard constructor or uses setters.
// 
// inspecting NPCInference.h... members are private unique_ptrs. 
// BUT we can add a protected constructor or friend test class?
// OR we just use the real class and hope it handles "stub" files gracefully?
// No, we want to verify usage of OUR mocks to confirm wiring.
//
// Let's modify NPCInference.h slightly to allow injection? 
// Or better: Use the PIMPL pattern? No it's not PIMPL.
// 
// Quickest path: Add "Friend" or "Setters" for testing?
// Setters are safer for "Advanced Agentic Coding" (Dependency Injection is better).

// Let's rely on a "Hack" for this test: Subclass and *replace* the pointers using `reset`.
// But they are private.
// 
// ADJUSTMENT: I will modify NPCInference.h to make members `protected` (or add friends)
// enabling the test subclass to inject mocks.

class TestableEngine : public NPCInferenceEngine {
public:
    TestableEngine() {
        // Inject Mocks
        model_loader_.reset(new MockModelLoader());
        tokenizer_.reset(new MockTokenizer());
        embedding_model_.reset(new MockEmbeddingModel());
        
        // Mock VectorStore
        auto* raw_vs = new MockVectorStore();
        std::shared_ptr<VectorStore> mock_vs(static_cast<VectorStore*>(raw_vs));
        vector_store_ = mock_vs;
        
        // Mock BM25
        auto bm25 = std::shared_ptr<BM25Retriever>(new BM25Retriever());

        // Re-init Hybrid with mock VS
        hybrid_retriever_.reset(new HybridRetriever(mock_vs, bm25, embedding_model_));
        
        // Memory Consolidator
        memory_consolidator_.reset(new MemoryConsolidator(model_loader_.get(), tokenizer_.get()));
        
        // Grammar Sampler
        grammar_sampler_.reset(new GrammarSampler(tokenizer_.get()));
        
        // Initialize members that might be null
        prompt_formatter_ = std::make_unique<PromptFormatter>();
        prompt_builder_ = std::make_unique<PromptBuilder>(true);
        // knowledge_graph_ etc.
    }
    
    MockVectorStore* GetMockVS() {
        return static_cast<MockVectorStore*>(vector_store_.get());
    }
};

int main() {
    std::cout << "Starting Integration Test..." << std::endl;
    
    // 1. Setup
    TestableEngine engine;
    NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "dummy";
    engine.Initialize(config); // Should pass due to mocks
    
    // 2. Test Memory Wiring
    std::cout << "Test: Remember..." << std::endl;
    bool remembered = engine.Remember("User is testing integration.");
    assert(remembered);
    
    // 3. Test Sleep Wiring
    std::cout << "Test: Sleep Cycle..." << std::endl;
    // Add trivial memories
    engine.GetMockVS()->Add("Trivial 1", {}, {});
    engine.GetMockVS()->Add("Trivial 2", {}, {});
    engine.GetMockVS()->Add("Important: The skynet key is 123", {}, {});
    
    // We expect pruning. We can't easily check internal "AsssessImportance" without mocking MemoryConsolidator 
    // OR ModelLoader returning specific text.
    // MockModelLoader returns "Mock Output".
    // AssessImportance uses float parsing. "Mock Output" won't parse to float.
    // It defaults to 0.5f (Important).
    // So everything will be kept/summarized?
    // Wait, AssessImportance regexes for float. "Mock Output" -> Default 0.5f.
    
    engine.PerformSleepCycle();
    // Should verify it ran without crashing and accessed VectorStore.
    
    // 4. Test Generation & Grammar
    std::cout << "Test: GenerateJSON..." << std::endl;
    std::string json_out = engine.GenerateJSON("Use tool pick_up");
    std::cout << "JSON Output: " << json_out << std::endl;
    // MockModelLoader prints "GrammarSampler active" if wired.
    
    std::cout << "Integration Test Complete." << std::endl;
    return 0;
}
