#include <iostream>
#include <vector>
#include "NPCInference.h"

using namespace NPCInference;

int main() {
    NPCInferenceEngine engine;
    
    NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "models/phi3-mini";
    config.enable_planner = true;
    config.enable_reflection = true; // Phase 2: Reflection Engine
    
    std::cout << "Initializing NPC Engine with Reflection Engine..." << std::endl;
    if (!engine.Initialize(config)) {
        std::cerr << "Failed to initialize engine." << std::endl;
        return 1;
    }

    nlohmann::json npcData = {
        {"name", "Guard Kael"},
        {"persona_vi", "Bạn là một lính gác nghiêm khắc nhưng công bằng. Bạn nói chuyện ngắn gọn và chuyên nghiệp."},
        {"persona_en", "You are a strict but fair guard. You speak briefly and professionally."}
    };

    nlohmann::json gameState = {
        {"scenario_plot", "Người chơi lảng vảng gần khu vực cấm lúc nửa đêm."},
        {"location", "Cổng thành phía Bắc"},
        {"conversation_id", "test_reflection_001"}
    };

    std::string playerInput = "Tôi chỉ đang đi dạo thôi mà, có sao đâu?";
    
    std::cout << "\n--- Testing Reflection Engine (Self-Correction) ---" << std::endl;
    std::cout << "User: " << playerInput << std::endl;
    
    std::string response = engine.GenerateWithState(playerInput, gameState, false);
    
    std::cout << "\nFinal Refined Output:" << std::endl;
    std::cout << response << std::endl;
    
    std::cout << "\n--------------------------------------------------" << std::endl;
    
    engine.GetProfiler().PrintSummary();

    return 0;
}
