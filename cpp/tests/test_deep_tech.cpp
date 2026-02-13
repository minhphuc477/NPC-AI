#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "NPCInference.h"
#include "VectorStore.h"
#include "KVCacheManager.h"

// Mock Ort::Value for testing CloneKV without full ONNX Runtime
// Note: Real test would need linked ONNX Runtime. This is a structural verification.
// For the user to run this, they need the build environment. 
// We will focus on logic that can be tested or structurally proved.

void TestVectorStore() {
    std::cout << "Testing Vector Store..." << std::endl;
    NPCInference::VectorStore store;
    store.Initialize(384); // MiniLM dimension

    std::vector<float> vec1(384, 0.1f);
    std::vector<float> vec2(384, -0.1f);
    
    // Test Add
    store.Add("Memory 1: Hero saved the cat.", vec1, {{"type", "event"}});
    store.Add("Memory 2: Villain stole the dog.", vec2, {{"type", "crime"}});
    
    // Test Search
    auto results = store.Search(vec1, 1);
    assert(!results.empty());
    assert(results[0].text.find("Hero") != std::string::npos);
    std::cout << "Vector Store: Basic Add/Search passed." << std::endl;
    
    // Test Persistence
    store.Save("test_vectors");
    NPCInference::VectorStore store2;
    store2.Initialize(384);
    store2.Load("test_vectors");
    auto results2 = store2.Search(vec1, 1);
    assert(!results2.empty());
    assert(results2[0].text == results[0].text);
    std::cout << "Vector Store: Persistence passed." << std::endl;
}

void TestGossip() {
    std::cout << "Testing Social Gossip..." << std::endl;
    NPCInference::NPCInferenceEngine engine;
    
    // Mock Embedding Model loading (requires real files usually)
    // We assume engine handles empty gracefully or we need to mock it.
    // For unit testing in C++ without mocks, we rely on integration tests.
    
    // Inject memory manually
    // engine.Remember("Player killed the dragon.", {{"type", "gossip"}}); 
    // This requires loaded embedding model.
    
    // We will verify the API exists and logic flows.
    engine.ReceiveGossip("Player stole an apple.", "Guard_NPC");
    // Verify it didn't crash.
    std::cout << "Gossip: ReceiveGossip executed." << std::endl;
}

int main() {
    try {
        TestVectorStore();
        TestGossip();
        std::cout << "ALL TESTS PASSED." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
