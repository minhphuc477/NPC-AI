// NPCInference.h - Main inference engine combining all components

#pragma once

#include "PromptFormatter.h"
#include "ModelLoader.h"
#include "PythonBridge.h"
#include "BehaviorTree.h"
#include "PromptBuilder.h"
#include "Tokenizer.h"
#include "VectorStore.h"
#include "EmbeddingModel.h"
#include <string>
#include <memory>
#include <optional>
#include <map>

namespace NPCInference {

/**
 * Main NPC Inference Engine
 * Combines prompt formatting, tokenization, and model inference
 * Matches the Python NPCInferenceEngine interface
 */
class NPCInferenceEngine {
public:
    NPCInferenceEngine();
    ~NPCInferenceEngine();
    
    /**
     * Load model and adapter
     * @param model_path Path to model
     * @param adapter_path Path to adapter (if any)
     * @param use_cuda Whether to use CUDA
     * @param bridge_mode Whether to use Python bridge (fallback)
     * @return true if loaded successfully
     */
    bool LoadModel(
        const std::string& model_path,
        const std::string& adapter_path = "",
        bool use_cuda = true,
        bool bridge_mode = false
    );
    
    /**
     * Load model using Python Bridge specifically
     */
    bool LoadWithBridge(
        const std::string& python_executable,
        const std::string& script_path,
        const std::string& model_path
    );

    /**
     * Generate NPC response from prompt
     * 
     * @param prompt Formatted prompt string
     * @param npc_name NPC name for logging
     * @return Generated response text
     */

    // Initialize the engine


    // Persistence
    bool SaveState(const std::string& filepath);
    bool LoadState(const std::string& filepath);

    // Update Game State and Tick Behavior Tree
    std::string UpdateState(const nlohmann::json& gameState);

    // Generate response based on current state
    std::string Generate(const std::string& prompt);
    std::string Generate(const std::string& prompt, const std::string& npc_name);

    // Convenience method for chat format
    std::string GenerateFromContext(const std::string& persona, const std::string& npc_id, const std::string& scenario, const std::string& player_input);

    /**
     * Store new memory in VectorStore
     * @param text content to remember
     * @param metadata additional info (e.g. timestamp, type)
     * @return true if successful
     */
    bool Remember(const std::string& text, const std::map<std::string, std::string>& metadata = {});

    /**
     * Save vector store to disk
     * @return true if successful
     */
    bool SaveMemory(); // Uses loaded path or default
    
    // Helper to formatting prompt
    std::string FormatPrompt(const std::string& system, const std::string& name, const std::string& context, const std::string& question);

    // Tokenize text to IDs
    std::vector<int64_t> Tokenize(const std::string& text);
    
    // Decode token IDs to text
    std::string Decode(const std::vector<int64_t>& token_ids);

    bool IsReady() const { return ready_; }

private:
    bool ready_ = false;
    bool bridge_mode_ = false;
    
    // Members
    std::unique_ptr<ModelLoader> model_loader_;
    std::unique_ptr<PromptFormatter> prompt_formatter_;
    std::unique_ptr<PythonBridge> python_bridge_;
    std::unique_ptr<Tokenizer> tokenizer_;
    
    // New Components
    std::shared_ptr<NPCBehavior::Node> behavior_tree_;
    std::unique_ptr<PromptBuilder> prompt_builder_;
    std::unique_ptr<VectorStore> vector_store_;
    std::unique_ptr<EmbeddingModel> embedding_model_;
    
public:
    // Configuration Struct
    struct InferenceConfig {
        std::string model_dir;
        std::string embedding_model_name = "embedding.onnx";
        std::string tokenizer_embedding_path = "tokenizer_embedding/sentencepiece.bpe.model";
        float rag_threshold = 0.6f;
        bool use_cuda = true;
        int num_threads = 4;
    };

    struct GenerationResult {
        std::string text;
        std::optional<std::string> tool_call; // JSON string of tool call
    };

    // Initialize with Config
    bool Initialize(const InferenceConfig& config);
    // Legacy Init (keeps compatibility)
    bool Initialize(const std::string& modelPath);

    // Output Parser
    GenerationResult ParseOutput(const std::string& raw_output);

private:
    // Internal State
    nlohmann::json current_state_;
    std::string current_action_ = "Idle";
    InferenceConfig config_;

public:
    void SetRagThreshold(float threshold) { config_.rag_threshold = threshold; }
};

} // namespace NPCInference
