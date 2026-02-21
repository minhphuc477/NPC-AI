#include "OrtEnvironmentManager.h"
#include "NPCLogger.h"

namespace NPCInference {

OrtEnvironmentManager& OrtEnvironmentManager::Instance() {
    static OrtEnvironmentManager instance;
    return instance;
}

OrtEnvironmentManager::OrtEnvironmentManager() {
    NPCLogger::Info("Initializing Global ONNX Runtime Environment...");
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NPCInference_Global");
    } catch (const std::exception& e) {
        NPCLogger::Error("Failed to initialize ONNX Runtime Environment: " + std::string(e.what()));
    }
}

Ort::Env& OrtEnvironmentManager::GetEnv() {
    return *env_;
}

} // namespace NPCInference
