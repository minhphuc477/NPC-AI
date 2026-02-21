#pragma once

#include <string>
#include <future>
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
     * Generate text completion synchronously with stream events
     * @param prompt The prompt to send
     * @param on_token_callback Callback fired when a new token arrives
     * @param on_action_callback Callback fired when a complete *action* string is parsed out
     * @param max_tokens Maximum tokens to generate
     * @param temperature Sampling temperature (0.0-1.0)
     * @return Full generated text (blocks until stream ends)
     */
    std::string GenerateStream(
        const std::string& prompt,
        std::function<void(const std::string&)> on_token_callback,
        std::function<void(const std::string&)> on_action_callback,
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
    
    /**
     * Cancel the ongoing generate request on background threads
     */
    void Cancel();

private:
    std::string model_name_;
    std::string base_url_;
    
    // Cancellation atomic flag
    std::atomic<bool> cancel_flag_{false};
    
    // Windows specifically needs the WinHttp Request handle exposed to cancel reliably
#ifdef _WIN32
    void* active_request_handle_ = nullptr;
    std::mutex handle_mutex_;
#endif
    
    std::string HttpPost(const std::string& url, const std::string& json_data, std::atomic<bool>* local_cancel = nullptr, std::function<void(const std::string&)> on_chunk_received = nullptr);
};

} // namespace NPCInference
