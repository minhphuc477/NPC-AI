#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace NPCInference {

/**
 * VisionLoader - Vision-Language Model Integration
 * 
 * Status: FULL IMPLEMENTATION
 * 
 * Supports:
 * - Phi-3-Vision ONNX model loading
 * - Image preprocessing (resize, normalize)
 * - Vision encoder inference
 * - Scene description generation
 * 
 * Usage:
 *   VisionLoader loader;
 *   loader.Load("models/vision_encoder_int8.onnx");
 *   auto result = loader.AnalyzeSceneDetailed(image_data, width, height);
 */
class VisionLoader {
public:
    VisionLoader();
    ~VisionLoader();

    /**
     * Configuration for vision model
     */
    struct VisionConfig {
        std::string model_path = "models/vision_encoder_int8.onnx";
        int image_size = 336;  // Phi-3-Vision input size
        bool use_cuda = true;
        int num_threads = 2;
    };

    /**
     * Object detection result
     */
    struct ObjectInfo {
        std::string label;
        float confidence;
        std::string description;
    };

    /**
     * Detailed scene analysis result
     */
    struct SceneAnalysis {
        std::string description;
        float confidence;
        std::vector<std::string> detected_objects;
        std::vector<float> object_confidences;
    };

    /**
     * Load vision model from ONNX file
     * @param model_path Path to vision_encoder.onnx
     * @return true if successful
     */
    bool Load(const std::string& model_path);

    /**
     * Analyze scene from raw image data
     * @param image_data RGB image data (HWC format)
     * @param width Image width
     * @param height Image height
     * @return Scene description string
     */
    std::string AnalyzeScene(const std::vector<uint8_t>& image_data, int width, int height) const;

    /**
     * Detailed scene analysis with confidence scores
     * @param image_data RGB image data
     * @param width Image width
     * @param height Image height
     * @return SceneAnalysis with description and detected objects
     */
    SceneAnalysis AnalyzeSceneDetailed(const std::vector<uint8_t>& image_data, int width, int height) const;

    /**
     * Analyze scene from embeddings
     */
    bool AnalyzeSceneDetailed(const std::vector<float>& embeddings, std::vector<ObjectInfo>& out_objects) const;

    /**
     * Check if model is loaded
     */
    bool IsLoaded() const { return loaded_; }

private:
    // Image preprocessing
    std::vector<float> PreprocessImage(const std::vector<uint8_t>& image_data, int width, int height) const;
    std::vector<float> ResizeImage(const std::vector<uint8_t>& data, int w, int h, int target_size) const;
    std::vector<float> NormalizeImage(const std::vector<float>& pixels) const;
    
    // Inference
    std::vector<float> RunVisionEncoder(const std::vector<float>& preprocessed) const;
    std::string DecodeEmbeddings(const std::vector<float>& embeddings) const;

    // ONNX Runtime
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    bool loaded_ = false;
    
    // Model I/O names
    std::string input_name_;
    std::string output_name_;
};

} // namespace NPCInference
