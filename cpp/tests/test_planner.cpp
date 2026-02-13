#include <iostream>
#include <vector>
#include "NPCInference.h"

using namespace NPCInference;

int main() {
    NPCInferenceEngine engine;
    
    NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "C:/Users/MPhuc/Desktop/NPC AI/models/phi3_onnx";
    config.enable_planner = true; // Phase 2: Cognitive Evolution
    
    std::cout << "Initializing NPC Engine with Planner-Executor split..." << std::endl;
    if (!engine.Initialize(config)) {
        std::cerr << "Failed to initialize engine." << std::endl;
        return 1;
    }

    nlohmann::json npcData = {
        {"name", "Lyra"},
        {"persona_vi", "Bạn là một học giả thông thái sống trong thư viện cổ. Bạn luôn suy nghĩ kỹ trước khi trả lời."},
        {"persona_en", "You are a wise scholar living in an ancient library. You always think carefully before answering."}
    };

    nlohmann::json gameState = {
        {"scenario_plot", "Người chơi vừa hỏi về một cuốn sách bị mất tích."},
        {"location", "Thư viện cổ"},
        {"conversation_id", "test_planner_001"}
    };

    std::string playerInput = "Bạn có biết cuốn 'Biên niên sử Antigravity' ở đâu không?";
    
    std::cout << "\n--- Testing Planner-Executor Split ---" << std::endl;
    std::cout << "User: " << playerInput << std::endl;
    
    std::string response = engine.GenerateWithState(playerInput, gameState, false);
    
    std::cout << "\nAssistant Output:" << std::endl;
    std::cout << response << std::endl;
    
    std::cout << "\n--------------------------------------" << std::endl;
    std::cout << "Verification Check:" << std::endl;
    if (response.find("thought") != std::string::npos || response.find("Suy nghĩ") != std::string::npos) {
        std::cout << "[PASS] Response contains the thinking phase." << std::endl;
    } else {
        std::cout << "[FAIL] Response missing internal monologue." << std::endl;
    }

    engine.GetProfiler().PrintSummary();

    return 0;
}
