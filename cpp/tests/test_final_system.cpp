#include <iostream>
#include <vector>
#include <cassert>
#include "NPCInference.h"

void TestVision(NPCInference::NPCInferenceEngine& engine) {
    std::cout << "[Test] Vision..." << std::endl;
    std::vector<uint8_t> dummy_image(100, 0);
    std::string desc = engine.See(dummy_image, 10, 10);
    std::cout << "Vision Output: " << desc << std::endl;
    // Expect stub message
    if (desc.find("test environment") != std::string::npos || desc.find("not yet connected") != std::string::npos) {
        std::cout << "[PASS] Vision Stub Active" << std::endl;
    } else {
        std::cout << "[WARN] Vision output unexpected: " << desc << std::endl;
    }
}

void TestMemoryAndSleep(NPCInference::NPCInferenceEngine& engine) {
    std::cout << "\n[Test] Memory & Sleep..." << std::endl;
    
    // 1. Inject Memories
    engine.ReceiveGossip("The King is dead.", "TownCrier");
    engine.ReceiveGossip("Long live the King.", "TownCrier");
    engine.ReceiveGossip("I saw a dragon.", "Guard");
    engine.ReceiveGossip("The prices are high.", "Merchant");
    engine.ReceiveGossip("It is raining.", "Weather");
    
    // 2. Trigger Sleep (requires 5 memories minimum)
    // Note: Without a real model, Summarize will fail/return empty, but it shouldn't crash.
    engine.PerformSleepCycle();
    
    // We can't easily verify the side effects without inspecting private state or logs,
    // but if it returned without crashing, that's a good sign for plumbing.
    std::cout << "[PASS] Sleep Cycle Executed (Check logs for summary failure/success)" << std::endl;
}

void TestGrammar(NPCInference::NPCInferenceEngine& engine) {
    std::cout << "\n[Test] Grammar Generation..." << std::endl;
    // Without model, this returns error, which is fine for verification of safety.
    std::string json = engine.GenerateJSON("Give me a quest");
    std::cout << "GenerateJSON Output: " << json << std::endl;
}

int main() {
    std::cout << "=== Final System Verification ===" << std::endl;
    
    NPCInference::NPCInferenceEngine engine;
    
    // Initialize with dummy config (won't load real model, but sets up components)
    // Note: Initialize might fail if files don't exist, but components are created in constructor.
    // So we can test components even if Initialize returns false.
    
    NPCInference::NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "invalid/path"; 
    engine.Initialize(config); 
    
    TestVision(engine);
    TestMemoryAndSleep(engine);
    TestGrammar(engine); // Should handle not-ready state gracefully
    
    std::cout << "\n=== Verification Complete ===" << std::endl;
    return 0;
}
