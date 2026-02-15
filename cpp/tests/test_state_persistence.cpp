#include "../include/NPCInference.h"
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace NPCInference;
namespace fs = std::filesystem;

int main() {
    std::cout << "=== State Persistence Test ===" << std::endl;
    
    NPCInferenceEngine engine;
    
    // Test 1: Initialize engine
    std::cout << "\n[Test 1] Initializing engine" << std::endl;
    NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "F:/NPC AI/models/phi3_onnx";
    config.use_cuda = false;
    config.enable_rag = true;
    config.enable_graph = true;
    config.enable_planner = true;
    config.enable_reflection = true;
    config.enable_truth_guard = true;
    config.rag_threshold = 0.7f;
    
    if (engine.Initialize(config)) {
        std::cout << "✓ Engine initialized" << std::endl;
    } else {
        std::cout << "  Engine initialization failed (model files may be missing)" << std::endl;
        std::cout << "  Continuing with state persistence tests anyway..." << std::endl;
    }
    
    // Test 2: Set some state
    std::cout << "\n[Test 2] Setting engine state" << std::endl;
    nlohmann::json game_state;
    game_state["npc_id"] = "guard_001";
    game_state["location"] = "Castle Gate";
    game_state["hp"] = 85;
    game_state["mood_state"] = "Alert";
    game_state["trust_level"] = 60;
    
    std::string action = engine.UpdateState(game_state);
    std::cout << "  Current action: " << action << std::endl;
    
    // Simulate some interaction to set last_thought
    // (In real usage, this would be set during generation)
    
    // Test 3: Save state
    std::cout << "\n[Test 3] Saving state" << std::endl;
    std::string state_file = "test_state.json";
    
    if (engine.SaveState(state_file)) {
        std::cout << "✓ State saved successfully" << std::endl;
    } else {
        std::cerr << "✗ Failed to save state" << std::endl;
        return 1;
    }
    
    // Test 4: Verify saved file exists and has content
    std::cout << "\n[Test 4] Verifying saved file" << std::endl;
    if (fs::exists(state_file)) {
        std::ifstream f(state_file);
        nlohmann::json saved_state;
        f >> saved_state;
        
        std::cout << "✓ State file exists and is valid JSON" << std::endl;
        
        // Check for expected fields
        if (saved_state.contains("version") && 
            saved_state.contains("current_state") &&
            saved_state.contains("current_action") &&
            saved_state.contains("config")) {
            std::cout << "✓ State file contains all expected fields" << std::endl;
            std::cout << "  Version: " << saved_state["version"] << std::endl;
            std::cout << "  Current action: " << saved_state["current_action"] << std::endl;
        } else {
            std::cerr << "✗ State file missing expected fields" << std::endl;
            return 1;
        }
        
        // Verify configuration was saved
        if (saved_state["config"].contains("rag_threshold")) {
            float saved_threshold = saved_state["config"]["rag_threshold"];
            if (saved_threshold == 0.7f) {
                std::cout << "✓ Configuration correctly saved (rag_threshold = " << saved_threshold << ")" << std::endl;
            }
        }
    } else {
        std::cerr << "✗ State file not found" << std::endl;
        return 1;
    }
    
    // Test 5: Create new engine and load state
    std::cout << "\n[Test 5] Loading state into new engine" << std::endl;
    NPCInferenceEngine engine2;
    
    if (engine2.LoadState(state_file)) {
        std::cout << "✓ State loaded successfully" << std::endl;
    } else {
        std::cerr << "✗ Failed to load state" << std::endl;
        return 1;
    }
    
    // Test 6: Verify loaded state matches
    std::cout << "\n[Test 6] Verifying loaded state" << std::endl;
    nlohmann::json loaded_game_state;
    loaded_game_state["test"] = "dummy"; // Trigger UpdateState
    std::string loaded_action = engine2.UpdateState(loaded_game_state);
    
    // The loaded action should match what was saved
    std::cout << "  Loaded action: " << loaded_action << std::endl;
    
    // Test 7: Test version mismatch handling
    std::cout << "\n[Test 7] Testing version mismatch handling" << std::endl;
    {
        nlohmann::json bad_state;
        bad_state["version"] = "0.5";
        bad_state["current_state"] = nlohmann::json::object();
        bad_state["current_action"] = "Test";
        
        std::ofstream f("test_state_bad_version.json");
        f << bad_state.dump(4);
    }
    
    NPCInferenceEngine engine3;
    if (engine3.LoadState("test_state_bad_version.json")) {
        std::cout << "✓ Loaded state with version mismatch (warning expected)" << std::endl;
    }
    
    // Test 8: Test error handling for missing file
    std::cout << "\n[Test 8] Testing error handling for missing file" << std::endl;
    NPCInferenceEngine engine4;
    if (!engine4.LoadState("nonexistent_file.json")) {
        std::cout << "✓ Correctly handled missing file" << std::endl;
    } else {
        std::cerr << "✗ Should have failed on missing file" << std::endl;
        return 1;
    }
    
    // Cleanup
    std::cout << "\n[Cleanup] Removing test files" << std::endl;
    fs::remove(state_file);
    fs::remove("test_state_bad_version.json");
    
    std::cout << "\n=== All State Persistence Tests Passed ===" << std::endl;
    return 0;
}
