#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>
#include <mutex>

namespace NPCInference {

class OrtEnvironmentManager {
public:
    static OrtEnvironmentManager& Instance();

    Ort::Env& GetEnv();

private:
    OrtEnvironmentManager();
    ~OrtEnvironmentManager() = default;

    std::unique_ptr<Ort::Env> env_;
    std::mutex mutex_;
};

} // namespace NPCInference
