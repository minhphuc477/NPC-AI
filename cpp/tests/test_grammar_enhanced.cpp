// test_grammar_enhanced.cpp - Test enhanced GrammarSampler with comprehensive validation

#include "GrammarSampler.h"
#include "Tokenizer.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace NPCInference;

void test_simple_object() {
    std::cout << "Test 1: Simple Object Generation..." << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.Load("models/tokenizer.model");
    
    GrammarSampler sampler(&tokenizer);
    
    // Simulate generating: {"tool": "get_time"}
    std::vector<std::string> tokens = {"{", "\"", "tool", "\"", ":", " ", "\"", "get_time", "\"", "}"};
    
    for (const auto& token : tokens) {
        auto ids = tokenizer.Encode(token);
        if (!ids.empty()) {
            sampler.AcceptToken(ids.back());
        }
    }
    
    std::string result = "{\"tool\": \"get_time\"}";
    assert(sampler.ValidateJSON(result));
    std::cout << "✓ Simple object validation passed" << std::endl;
}

void test_nested_object() {
    std::cout << "Test 2: Nested Object..." << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.Load("models/tokenizer.model");
    
    GrammarSampler sampler(&tokenizer);
    
    std::string result = "{\"tool\": \"search\", \"parameters\": {\"query\": \"test\"}}";
    assert(sampler.ValidateJSON(result));
    std::cout << "✓ Nested object validation passed" << std::endl;
}

void test_array_values() {
    std::cout << "Test 3: Array Values..." << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.Load("models/tokenizer.model");
    
    GrammarSampler sampler(&tokenizer);
    
    std::string result = "{\"items\": [\"a\", \"b\", \"c\"]}";
    assert(sampler.ValidateJSON(result));
    std::cout << "✓ Array validation passed" << std::endl;
}

void test_number_values() {
    std::cout << "Test 4: Number Values..." << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.Load("models/tokenizer.model");
    
    GrammarSampler sampler(&tokenizer);
    
    std::string result = "{\"count\": 42, \"price\": 19.99}";
    assert(sampler.ValidateJSON(result));
    std::cout << "✓ Number validation passed" << std::endl;
}

void test_boolean_null() {
    std::cout << "Test 5: Boolean and Null Values..." << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.Load("models/tokenizer.model");
    
    GrammarSampler sampler(&tokenizer);
    
    std::string result = "{\"active\": true, \"disabled\": false, \"data\": null}";
    assert(sampler.ValidateJSON(result));
    std::cout << "✓ Boolean/null validation passed" << std::endl;
}

void test_complex_tool_call() {
    std::cout << "Test 6: Complex Tool Call..." << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.Load("models/tokenizer.model");
    
    GrammarSampler sampler(&tokenizer);
    
    std::string result = R"({
        "tool": "execute_query",
        "parameters": {
            "query": "SELECT * FROM users WHERE age > 18",
            "limit": 100,
            "include_metadata": true,
            "filters": ["active", "verified"]
        }
    })";
    
    assert(sampler.ValidateJSON(result));
    std::cout << "✓ Complex tool call validation passed" << std::endl;
}

void test_invalid_json() {
    std::cout << "Test 7: Invalid JSON Detection..." << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.Load("models/tokenizer.model");
    
    GrammarSampler sampler(&tokenizer);
    
    // Missing closing brace
    std::string invalid1 = "{\"tool\": \"test\"";
    assert(!sampler.ValidateJSON(invalid1));
    
    // Missing quote
    std::string invalid2 = "{tool: \"test\"}";
    assert(!sampler.ValidateJSON(invalid2));
    
    // Trailing comma
    std::string invalid3 = "{\"tool\": \"test\",}";
    assert(!sampler.ValidateJSON(invalid3));
    
    std::cout << "✓ Invalid JSON correctly detected" << std::endl;
}

void test_state_transitions() {
    std::cout << "Test 8: State Transition Tracking..." << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.Load("models/tokenizer.model");
    
    GrammarSampler sampler(&tokenizer);
    
    // Initial state
    assert(sampler.GetState() == GrammarSampler::JsonState::WAITING_FOR_OPEN_BRACE);
    
    // After {
    auto ids = tokenizer.Encode("{");
    if (!ids.empty()) {
        sampler.AcceptToken(ids.back());
        assert(sampler.GetState() == GrammarSampler::JsonState::WAITING_FOR_KEY_QUOTE_START);
    }
    
    std::cout << "✓ State transitions working correctly" << std::endl;
}

void test_logit_filtering() {
    std::cout << "Test 9: Logit Filtering..." << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.Load("models/tokenizer.model");
    
    GrammarSampler sampler(&tokenizer);
    
    // Create dummy logits
    const int vocab_size = 32000;
    std::vector<float> logits(vocab_size, 0.0f);
    
    // At start, only { should be allowed
    sampler.FilterLogits(logits.data(), vocab_size);
    
    // Check that most tokens are masked
    int masked_count = 0;
    for (float logit : logits) {
        if (std::isinf(logit) && logit < 0) {
            masked_count++;
        }
    }
    
    assert(masked_count > vocab_size * 0.99); // At least 99% masked
    std::cout << "✓ Logit filtering working (masked " << masked_count << "/" << vocab_size << " tokens)" << std::endl;
}

void test_escape_sequences() {
    std::cout << "Test 10: Escape Sequences..." << std::endl;
    
    Tokenizer tokenizer;
    tokenizer.Load("models/tokenizer.model");
    
    GrammarSampler sampler(&tokenizer);
    
    std::string result = R"({"message": "He said \"Hello\""})";
    assert(sampler.ValidateJSON(result));
    
    std::string result2 = R"({"path": "C:\\Users\\Test"})";
    assert(sampler.ValidateJSON(result2));
    
    std::cout << "✓ Escape sequences handled correctly" << std::endl;
}

int main() {
    std::cout << "=== Enhanced GrammarSampler Test Suite ===" << std::endl << std::endl;
    
    try {
        test_simple_object();
        test_nested_object();
        test_array_values();
        test_number_values();
        test_boolean_null();
        test_complex_tool_call();
        test_invalid_json();
        test_state_transitions();
        test_logit_filtering();
        test_escape_sequences();
        
        std::cout << std::endl << "=== ALL TESTS PASSED ✓ ===" << std::endl;
        std::cout << "GrammarSampler enhancements verified successfully!" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
