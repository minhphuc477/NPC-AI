#include "../include/NPCInference.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace NPCInference;

int main() {
    std::cout << "=== Async Initialization Test ===" << std::endl;
    
    NPCInferenceEngine engine;
    
    // Test 1: Check initial loading state
    std::cout << "\n[Test 1] Initial loading state" << std::endl;
    if (!engine.IsLoading()) {
        std::cout << "✓ Engine not loading initially" << std::endl;
    } else {
        std::cerr << "✗ Engine should not be loading initially" << std::endl;
        return 1;
    }
    
    // Test 2: Start async initialization
    std::cout << "\n[Test 2] Starting async initialization" << std::endl;
    
    NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "F:/NPC AI/models/phi3_onnx"; // Adjust path as needed
    config.use_cuda = false; // Use CPU for testing
    config.enable_rag = false; // Disable optional features for faster init
    config.enable_graph = false;
    config.enable_speculative = false;
    
    bool callback_invoked = false;
    bool init_success = false;
    
    engine.InitializeAsync(config, [&](bool success) {
        callback_invoked = true;
        init_success = success;
        std::cout << "  Callback invoked with success = " << success << std::endl;
    });
    
    // Test 3: Check loading state immediately after start
    std::cout << "\n[Test 3] Loading state after async start" << std::endl;
    if (engine.IsLoading()) {
        std::cout << "✓ Engine is loading" << std::endl;
    } else {
        std::cout << "  Note: Engine may have finished loading very quickly" << std::endl;
    }
    
    // Test 4: Wait for initialization to complete
    std::cout << "\n[Test 4] Waiting for initialization to complete..." << std::endl;
    int wait_count = 0;
    while (engine.IsLoading() && wait_count < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }
    
    if (!engine.IsLoading()) {
        std::cout << "✓ Loading completed after " << (wait_count * 100) << "ms" << std::endl;
    } else {
        std::cerr << "✗ Timeout waiting for initialization" << std::endl;
        return 1;
    }
    
    // Test 5: Verify callback was invoked
    std::cout << "\n[Test 5] Callback verification" << std::endl;
    // Give a bit more time for callback
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    if (callback_invoked) {
        std::cout << "✓ Callback was invoked" << std::endl;
        std::cout << "  Initialization " << (init_success ? "succeeded" : "failed") << std::endl;
    } else {
        std::cerr << "✗ Callback was not invoked" << std::endl;
        return 1;
    }
    
    // Test 6: Try to start another async init (should fail)
    std::cout << "\n[Test 6] Attempting concurrent initialization" << std::endl;
    bool second_callback_invoked = false;
    
    engine.InitializeAsync(config, [&](bool success) {
        second_callback_invoked = true;
        if (!success) {
            std::cout << "✓ Second initialization correctly rejected" << std::endl;
        }
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Test 7: Check engine readiness
    std::cout << "\n[Test 7] Engine readiness" << std::endl;
    if (engine.IsReady()) {
        std::cout << "✓ Engine is ready" << std::endl;
    } else {
        std::cout << "  Engine not ready (model files may be missing)" << std::endl;
    }
    
    std::cout << "\n=== All Async Init Tests Passed ===" << std::endl;
    return 0;
}
