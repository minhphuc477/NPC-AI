#include "NPCInference.h"
#include <iostream>

using namespace NPCInference;

int main() {
    NPCInferenceEngine::InferenceConfig config;
    config.model_dir = "models/phi3_onnx_real";
    
    NPCInferenceEngine engine;
    if (!engine.Initialize(config)) {
        std::cerr << "Failed to initialize engine.\n";
        return 1;
    }
    
    std::string response = engine.GenerateFromContext("Persona", "ID", "Scenario", "Hello");
    std::cout << "Response:\n" << response << "\n";
    return 0;
}
