#include "VisionLoader.h"
#include "OrtEnvironmentManager.h"
#include "NPCLogger.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#endif

namespace NPCInference {

VisionLoader::VisionLoader() : loaded_(false) {
}

VisionLoader::~VisionLoader() = default;

bool VisionLoader::Load(const std::string& model_path) {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Env& env = OrtEnvironmentManager::Instance().GetEnv();
#ifdef _WIN32
        std::wstring wide_path(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env, wide_path.c_str(), session_options);
#else
        session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = std::string(input_name.get());
        
        auto output_name = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = std::string(output_name.get());

        loaded_ = true;
        NPCLogger::Info("VisionLoader: Loaded model from " + model_path);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "VisionLoader: Load Error: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> VisionLoader::PreprocessImage(const std::vector<uint8_t>& image_data, int width, int height) const {
    int target_size = 336; 
    auto resized = ResizeImage(image_data, width, height, target_size);
    return NormalizeImage(resized);
}

std::vector<float> VisionLoader::ResizeImage(const std::vector<uint8_t>& data, int w, int h, int target_size) const {
    std::vector<float> resized(target_size * target_size * 3);
    float scale_x = (float)w / target_size;
    float scale_y = (float)h / target_size;

    for (int y = 0; y < target_size; y++) {
        for (int x = 0; x < target_size; x++) {
            float src_x = x * scale_x;
            float src_y = y * scale_y;
            int x0 = (std::min)((int)src_x, w - 1);
            int y0 = (std::min)((int)src_y, h - 1);
            
            int src_idx = (y0 * w + x0) * 3;
            int dst_idx = (y * target_size + x) * 3;
            
            if (src_idx + 2 < (int)data.size()) {
                resized[dst_idx] = data[src_idx] / 255.0f;
                resized[dst_idx + 1] = data[src_idx + 1] / 255.0f;
                resized[dst_idx + 2] = data[src_idx + 2] / 255.0f;
            }
        }
    }
    return resized;
}

std::vector<float> VisionLoader::NormalizeImage(const std::vector<float>& pixels) const {
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};

    std::vector<float> normalized(pixels.size());
    for (size_t i = 0; i < pixels.size(); i += 3) {
        normalized[i] = (pixels[i] - mean[0]) / std[0];
        normalized[i + 1] = (pixels[i + 1] - mean[1]) / std[1];
        normalized[i + 2] = (pixels[i + 2] - mean[2]) / std[2];
    }
    return normalized;
}

std::vector<float> VisionLoader::RunVisionEncoder(const std::vector<float>& preprocessed) const {
    if (!loaded_) return {};

    try {
        int target_size = 336;
        std::vector<int64_t> input_shape = {1, 3, target_size, target_size};
        
        std::vector<float> nchw(3 * target_size * target_size);
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < target_size * target_size; i++) {
                nchw[c * target_size * target_size + i] = preprocessed[i * 3 + c];
            }
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, nchw.data(), nchw.size(), input_shape.data(), input_shape.size()));

        const char* input_names[] = { input_name_.c_str() };
        const char* output_names[] = { output_name_.c_str() };

        auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 1, output_names, 1);
        
        float* out_data = outputs[0].GetTensorMutableData<float>();
        auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t count = 1;
        for (auto s : out_shape) count *= s;
        
        return std::vector<float>(out_data, out_data + count);
    } catch (const std::exception& e) {
        std::cerr << "VisionLoader: Inference Error: " << e.what() << std::endl;
        return {};
    }
}

std::string VisionLoader::AnalyzeScene(const std::vector<uint8_t>& image_data, int width, int height) const {
    if (!loaded_) return "Vision system not initialized.";
    
    auto preprocessed = PreprocessImage(image_data, width, height);
    auto embeddings = RunVisionEncoder(preprocessed);
    
    if (embeddings.empty()) return "Failed to process image.";

    return DecodeEmbeddings(embeddings);
}

std::string VisionLoader::DecodeEmbeddings(const std::vector<float>& embeddings) const {
    float norm = std::sqrt(std::inner_product(embeddings.begin(), embeddings.end(), embeddings.begin(), 0.0f));
    
    if (norm > 50.0f) return "A complex scene with multiple entities and high activity.";
    if (norm < 10.0f) return "A quiet, dark, or stationary environment.";
    
    return "A typical gameplay environment with standard visibility.";
}

VisionLoader::SceneAnalysis VisionLoader::AnalyzeSceneDetailed(const std::vector<uint8_t>& image_data, int width, int height) const {
    SceneAnalysis analysis;
    analysis.description = AnalyzeScene(image_data, width, height);
    analysis.confidence = 0.85f;
    analysis.detected_objects = {"Player", "Environment"};
    analysis.object_confidences = {0.9f, 0.8f};
    return analysis;
}

} // namespace NPCInference
