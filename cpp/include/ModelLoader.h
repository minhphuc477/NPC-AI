// ModelLoader.h - Loads ONNX model and manages inference session

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

// Forward declare ONNX Runtime types
namespace Ort {
    class Env;
    class Session;
    class SessionOptions;
    struct Value; // For completeness if needed, though PIMPL handles it
}

namespace NPCInference {
    class KVCacheManager;
}
namespace NPCInference {

class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();
    
    /**
     * Load ONNX model from file
     * @param model_path Path to .onnx model file
     * @param use_cuda Whether to use CUDA acceleration
     * @return true if loaded successfully
     */
    bool LoadModel(const std::string& model_path, bool use_cuda = true, int num_threads = 4);
    
    /**
     * Run inference on tokenized input with KV-cache support
     * @param input_ids Token IDs from tokenizer
     * @param attention_mask Attention mask
     * @param max_new_tokens Maximum tokens to generate
     * @param conversation_id Unique ID for conversation (enables KV-cache persistence)
     * @param on_token_callback Optional callback for streaming tokens
     * @return Generated token IDs
     */
    std::vector<int64_t> Generate(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& attention_mask,
        int max_new_tokens = 150,
        const std::string& conversation_id = "",
        std::function<void(int64_t)> on_token_callback = nullptr
    );
    
    /**
     * Clear KV-cache for a specific conversation
     */
    void ClearCache(const std::string& conversation_id = "");
    
    /**
     * Get KV-cache statistics
     */
    void PrintCacheStats() const;
    
    /**
     * Check if model is loaded
     */
    bool IsLoaded() const { return is_loaded_; }

    // Configuration setters
    void SetTemperature(float temp) { temperature_ = temp; }
    void SetTopP(float top_p) { top_p_ = top_p; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    bool is_loaded_ = false;
    
    // Model configuration
    float temperature_ = 0.7f;
    float top_p_ = 0.9f;
    float repetition_penalty_ = 1.1f;

    // KV Cache (PIMPL to avoid exposing Ort::Value in header)
    struct KVCache;
    std::unique_ptr<KVCache> kv_cache_;
    
    // Persistent KV-cache manager
    std::shared_ptr<KVCacheManager> cache_manager_;
    
    // Helper for sampling
    int64_t SampleToken(float* logits, int64_t vocab_size);
};

} // namespace NPCInference
