// NPCInference.h - Main inference engine combining all components

#pragma once

// Standard Library
#include <cstdint>
#include <string>
#include <memory>
#include <optional>
#include <map>
#include <future>
#include <atomic>
#include <mutex>
#include <functional>

// Third-party
#include <nlohmann/json.hpp>

// New Subsystems
#include "BehaviorTree.h"
#include "PromptBuilder.h"
#include "VectorStore.h"
#include "EmbeddingModel.h"
#include "SimpleGraph.h"
#include "MemoryConsolidator.h"
#include "VisionLoader.h"
#include "GrammarSampler.h"
#include "ToolRegistry.h"
#include "HybridRetriever.h"
#include "PerformanceProfiler.h"
#include "ConversationManager.h"
#include "TemporalMemorySystem.h"
#include "SocialFabricNetwork.h"
#include "EmotionalContinuitySystem.h"
#include "PlayerBehaviorModeling.h"
#include "AmbientAwarenessSystem.h"

namespace NPCInference {

    using json = nlohmann::json;

    // Forward decls (kept for clarity/minimal dependency if needed)
    class Tokenizer;
    class ModelLoader;
    class PromptFormatter;
    class PythonBridge;

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
    std::string GenerateWithState(const std::string& prompt, nlohmann::json& state, const std::string& last_thought = "", bool is_json = false);
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
     * Learn semantic knowledge from text (Dynamic OIE -> Graph)
     */
    void Learn(const std::string& text);

    // === Advanced NPC Systems ===
    
    /** Get temporal memory system */
    TemporalMemorySystem* GetTemporalMemory() { return temporal_memory_.get(); }
    
    /** Get social fabric network */
    SocialFabricNetwork* GetSocialFabric() { return social_fabric_network_.get(); }
    
    /** Get emotional continuity system */
    EmotionalContinuitySystem* GetEmotionalContinuity() { return emotional_continuity_system_.get(); }
    
    /** Get player behavior modeling system */
    PlayerBehaviorModeling* GetPlayerBehaviorModeling() { return player_behavior_modeling_.get(); }
    
    /** Get ambient awareness system */
    AmbientAwarenessSystem* GetAmbientAwareness() { return ambient_awareness_system_.get(); }
    
    /**
     * Build context from all advanced systems for NPC response generation
     * @param npc_id The NPC's identifier
     * @param query Current conversation query
     * @return JSON context with memories, relationships, and emotions
     */
    nlohmann::json BuildAdvancedContext(const std::string& npc_id, const std::string& query);

    // Get conversation manager for real-time chat
    ConversationManager* GetConversationManager() { return conversation_manager_.get(); }
    
    // === Real-Time Conversation API ===
    
    /**
     * Start a new conversation session
     * @param npc_name Name of the NPC
     * @param player_name Name of the player
     * @return session_id for this conversation
     */
    std::string StartConversation(const std::string& npc_name, const std::string& player_name);
    
    /**
     * Send a message in an active conversation
     * @param session_id Active conversation session
     * @param user_message Player's message
     * @return NPC's response
     */
    std::string Chat(const std::string& session_id, const std::string& user_message);
    
    /**
     * End conversation and trigger memory consolidation
     * @param session_id Session to close
     */
    void EndConversation(const std::string& session_id);

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
    std::unique_ptr<ConversationManager> conversation_manager_; // Real-time chat
    std::unique_ptr<TemporalMemorySystem> temporal_memory_; // Temporal memory with decay
    std::unique_ptr<SocialFabricNetwork> social_fabric_network_; // Social dynamics network
    std::unique_ptr<EmotionalContinuitySystem> emotional_continuity_system_; // Persistent emotions
    std::unique_ptr<PlayerBehaviorModeling> player_behavior_modeling_; // Player behavior tracking
    std::unique_ptr<AmbientAwarenessSystem> ambient_awareness_system_; // Event inference

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
        bool enable_python_bridge = false; // Fallback to Python Bridge
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
    mutable std::mutex state_mutex_;  // Thread safety for shared state
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
