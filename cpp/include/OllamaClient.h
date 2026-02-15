#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace NPCInference {

/**
 * Simple Ollama API Client
 * Calls Ollama's HTTP API at localhost:11434
 */
class OllamaClient {
public:
    OllamaClient(const std::string& model_name = "phi3:mini", 
                 const std::string& base_url = "http://localhost:11434");
    ~OllamaClient() = default;
    
    /**
     * Generate text completion
     * @param prompt The prompt to send
     * @param max_tokens Maximum tokens to generate
     * @param temperature Sampling temperature (0.0-1.0)
     * @return Generated text
     */
    std::string Generate(
        const std::string& prompt,
        int max_tokens = 150,
        float temperature = 0.7f
    );
    
    /**
     * Check if Ollama is running and model is available
     * @return true if ready
     */
    bool IsReady();
    
    /**
     * Set the model to use
     * @param model_name Model name (e.g., "phi3:mini", "llama2", etc.)
     */
    void SetModel(const std::string& model_name) { model_name_ = model_name; }
    
private:
    std::string model_name_;
    std::string base_url_;
    
    std::string HttpPost(const std::string& url, const std::string& json_data);
};

} // namespace NPCInference
