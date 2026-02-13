#include "GrammarSampler.h"
#include "Tokenizer.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <limits>
#include <algorithm>

// Mock Tokenizer
class MockTokenizer : public NPCInference::Tokenizer {
public:
    virtual std::vector<int64_t> Encode(const std::string& text) override {
        // Simple mapping for test
        if (text == "{") return {10};
        if (text == "}") return {11};
        if (text == "\"") return {12};
        if (text == ":") return {13};
        if (text == ",") return {14};
        return {99};
    }
    
    virtual std::string Decode(const std::vector<int64_t>& ids) override {
        if (ids.empty()) return "";
        int64_t id = ids[0];
        if (id == 10) return "{";
        if (id == 11) return "}";
        if (id == 12) return "\"";
        if (id == 13) return ":";
        if (id == 14) return ",";
        return "?";
    }
    
    virtual bool Load(const std::string& model_path) override { return true; }
    virtual bool IsLoaded() const override { return true; }
    virtual int GetVocabSize() const override { return 1000; }
    virtual int GetEOSId() const override { return 11; }
};

void TestGrammarStrictness() {
    MockTokenizer tokenizer;
    NPCInference::GrammarSampler sampler(&tokenizer);
    
    // reset
    sampler.Reset(); 
    // State: WAITING_FOR_OPEN_BRACE
    
    int vocab_size = 100;
    std::vector<float> logits(vocab_size, 0.0f);
    
    // Filter
    sampler.FilterLogits(logits.data(), vocab_size);
    
    // Assert: Only ID 10 ({) should be non-negative (or not -inf)
    // Actually we check if 10 is > -inf and others are -inf
    
    float val_10 = logits[10];
    float val_other = logits[50];
    
    std::cout << "Logit[{]: " << val_10 << std::endl;
    std::cout << "Logit[other]: " << val_other << std::endl;
    
    assert(val_10 > -1000.0f); // It should be 0.0 or boosted
    assert(val_other == -std::numeric_limits<float>::infinity());
    
    std::cout << "State 1 (Open Brace) Passed." << std::endl;
    
    // Advance state
    sampler.AcceptToken(10); // {
    // State: WAITING_FOR_KEY_QUOTE_START
    
    // Reset logits
    std::fill(logits.begin(), logits.end(), 0.0f);
    sampler.FilterLogits(logits.data(), vocab_size);
    
    // Allowed: " (12) or } (11)
    assert(logits[12] > -1000.0f);
    assert(logits[11] > -1000.0f);
    assert(logits[10] == -std::numeric_limits<float>::infinity());
    
    std::cout << "State 2 (Key Start) Passed." << std::endl;
    
    // Advance with "
    sampler.AcceptToken(12);
    // State: INSIDE_KEY
    
    // Filter
    std::fill(logits.begin(), logits.end(), 0.0f);
    sampler.FilterLogits(logits.data(), vocab_size);
    
    // Should NOT mask everything (allow free gen)
    assert(logits[50] == 0.0f); 
    
    std::cout << "State 3 (Inside Key) Passed." << std::endl;
    
    std::cout << "SUCCESS: Grammar logic verified." << std::endl;
}

int main() {
    try {
        TestGrammarStrictness();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
