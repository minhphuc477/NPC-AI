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
#include "SimpleGraph.h"
#include "HybridRetriever.h" // Added Phase 2
#include "ToolRegistry.h" // Added for tool execution
#include "PerformanceProfiler.h" // Added
#include <cstdint>
#include <string>
#include <memory>
#include <optional>
#include <map>
#include <future>
#include <atomic>
#include <functional>
#include <nlohmann/json.hpp>

namespace NPCInference {

    // Forward decls
    class Tokenizer;
    class ModelLoader;
    class PromptFormatter;
    class PythonBridge;
    class PromptBuilder;
    class VectorStore;
    class EmbeddingModel;
    class SimpleGraph;
    class MemoryConsolidator;
    class VisionLoader;
    class GrammarSampler;

    struct GenerationResult {
        std::string text;
        std::optional<std::string> tool_call; // JSON string of tool call
    };

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

    // Generate response based on current state (shared)
    std::string Generate(const std::string& prompt);
    // Thread-safe internal generation with state snapshot
    std::string GenerateWithState(const std::string& prompt, const nlohmann::json& state, bool is_json = false);
    std::string Generate(const std::string& prompt, const std::string& npc_name);
    
    // Phase 12: Structured Generation
    std::string GenerateJSON(const std::string& prompt);

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
     * Social Gossip System
     * Extract interesting memories to share with other NPCs
     */
    std::string ExtractGossip();
    void ReceiveGossip(const std::string& gossip_text, const std::string& source_npc);

    /**
     * @return true if successful
     */
    bool SaveMemory(); // Uses loaded path or default
    
    // Phase 10: Sleep Mode
    void PerformSleepCycle();

    // Phase 11: Vision
    std::string See(const std::vector<uint8_t>& image_data, int width, int height);

    // Tokenize text to IDs
    std::vector<int64_t> Tokenize(const std::string& text);
    std::string Decode(const std::vector<int64_t>& token_ids);

private:    // Helper to formatting prompt
    std::string FormatPrompt(const std::string& system, const std::string& name, const std::string& context, const std::string& question);
    
    // Phase 12: Tool Execution
    std::string ExecuteAction(const std::string& tool_call_json);

    // Context Parsing
    // ParseOutput moved to public



protected:
    bool ready_ = false;
    bool bridge_mode_ = false;
    
    // Members
    std::unique_ptr<ModelLoader> model_loader_;
    std::unique_ptr<ModelLoader> draft_model_loader_; // New
    std::unique_ptr<PromptFormatter> prompt_formatter_;
    std::unique_ptr<PythonBridge> python_bridge_;
    std::unique_ptr<Tokenizer> tokenizer_;
    
    // New Components
    std::shared_ptr<NPCBehavior::Node> behavior_tree_;
    std::unique_ptr<PromptBuilder> prompt_builder_;
    std::shared_ptr<VectorStore> vector_store_;
    std::shared_ptr<EmbeddingModel> embedding_model_;
    std::unique_ptr<SimpleGraph> knowledge_graph_;
    std::unique_ptr<MemoryConsolidator> memory_consolidator_; 
    std::unique_ptr<VisionLoader> vision_loader_;
    std::shared_ptr<GrammarSampler> grammar_sampler_; // Phase 12
    std::unique_ptr<ToolRegistry> tool_registry_;    // Phase 12
    std::unique_ptr<HybridRetriever> hybrid_retriever_; // Added Phase 2
    std::unique_ptr<PerformanceProfiler> profiler_; // Added

public:
    // Configuration Struct
    struct InferenceConfig {
        std::string model_dir;
        std::string draft_model_dir; // New
        std::string embedding_model_name = "embedding.onnx";
        std::string tokenizer_embedding_path = "tokenizer_embedding/sentencepiece.bpe.model";
        float rag_threshold = 0.6f;
        bool use_cuda = true;
        int num_threads = 4;
        
        // Ablation Toggles
        bool enable_rag = true;
        bool enable_graph = true;
        bool enable_speculative = true;
        bool enable_grammar = true;
        bool enable_planner = true;
        bool enable_reflection = true;
        bool enable_truth_guard = true; // Phase 2: Neuro-symbolic Truth Guard
    };


    // Initialize with Config
    bool Initialize(const InferenceConfig& config);
    // Asynchronous Initialization
    void InitializeAsync(const InferenceConfig& config, std::function<void(bool)> callback = nullptr);
    
    // Legacy Init (keeps compatibility)
    bool Initialize(const std::string& modelPath);

    // Check loading status
    bool IsLoading() const;

    // Output Parser
    GenerationResult ParseOutput(const std::string& raw_output);

    // Check ready status
    bool IsReady() const { return ready_; }

private:
    // Internal State
    nlohmann::json current_state_;
    std::string current_action_ = "Idle";
    std::string last_thought_ = "";
    InferenceConfig config_;
    
    // Async State
    std::future<bool> loading_future_;
    std::atomic<bool> is_loading_{false};

public:
    void SetRagThreshold(float threshold) { config_.rag_threshold = threshold; }
    
    // Performance Profiling
    PerformanceProfiler& GetProfiler() { return *profiler_; }
};

} // namespace NPCInference
