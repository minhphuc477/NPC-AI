#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <cmath>
#include <mutex>
#include <nlohmann/json.hpp>

namespace NPCInference {

/**
 * Temporal Memory System - Biologically-Inspired Memory with Decay
 * 
 * Features:
 * - Episodic Memory: Specific events with timestamps and emotional context
 * - Semantic Memory: General knowledge extracted from episodes
 * - Temporal Decay: Memories fade over time (configurable curve)
 * - Consolidation: Important episodes become semantic knowledge
 * - Emotional Weighting: Emotionally charged events remembered longer
 */

struct EpisodicMemory {
    std::string event_id;
    std::string description;
    int64_t timestamp;           // Unix timestamp
    float emotional_valence;     // -1 (negative) to +1 (positive)
    float emotional_arousal;     // 0 (calm) to 1 (intense)
    float importance;            // 0 to 1 (user-defined or auto-calculated)
    std::vector<std::string> participants;  // Who was involved
    std::string location;
    std::map<std::string, std::string> metadata;
    
    // Calculated fields
    float current_strength = 1.0f;  // Decays over time
    int retrieval_count = 0;        // Strengthens with retrieval
    
    EpisodicMemory() : timestamp(0), emotional_valence(0.0f), 
                      emotional_arousal(0.0f), importance(0.5f) {}
};

struct SemanticMemory {
    std::string concept_id;
    std::string knowledge;       // General fact or belief
    float confidence;            // 0 to 1
    std::vector<std::string> source_episodes;  // Which episodes support this
    int64_t first_learned;       // When first formed
    int64_t last_updated;        // When last reinforced
    
    SemanticMemory() : confidence(0.5f), first_learned(0), last_updated(0) {}
};

class TemporalMemorySystem {
public:
    TemporalMemorySystem();
    ~TemporalMemorySystem() = default;
    
    // === Episodic Memory Management ===
    
    /**
     * Add a new episodic memory
     * @param description What happened
     * @param emotional_valence How positive/negative (-1 to 1)
     * @param emotional_arousal How intense (0 to 1)
     * @param importance How important (0 to 1, auto-calculated if 0)
     * @param participants Who was involved
     * @param location Where it happened
     * @return Memory ID
     */
    std::string AddEpisode(
        const std::string& description,
        float emotional_valence = 0.0f,
        float emotional_arousal = 0.5f,
        float importance = 0.0f,  // 0 = auto-calculate
        const std::vector<std::string>& participants = {},
        const std::string& location = ""
    );
    
    /**
     * Retrieve episodic memories with temporal decay applied
     * @param query Search query
     * @param max_results Maximum number of results
     * @param min_strength Minimum strength threshold (after decay)
     * @return Matching memories, sorted by relevance * strength
     */
    std::vector<EpisodicMemory> RetrieveEpisodes(
        const std::string& query,
        int max_results = 10,
        float min_strength = 0.1f
    );
    
    /**
     * Get all episodes involving a specific entity
     */
    std::vector<EpisodicMemory> GetEpisodesWithEntity(
        const std::string& entity_name,
        int max_results = 20
    );
    
    // === Semantic Memory Management ===
    
    /**
     * Add or update semantic knowledge
     * @param knowledge The general fact or belief
     * @param confidence How confident (0 to 1)
     * @param source_episode Optional episode that supports this
     * @return Concept ID
     */
    std::string AddSemanticKnowledge(
        const std::string& knowledge,
        float confidence = 0.7f,
        const std::string& source_episode = ""
    );
    
    /**
     * Retrieve semantic knowledge
     */
    std::vector<SemanticMemory> RetrieveSemanticKnowledge(
        const std::string& query,
        int max_results = 5
    );
    
    // === Memory Consolidation ===
    
    /**
     * Consolidate episodic memories into semantic knowledge
     * This is the "sleep" process - run periodically
     * 
     * Algorithm:
     * 1. Find clusters of related episodes
     * 2. Extract common patterns
     * 3. Form semantic beliefs
     * 4. Strengthen important episodes, let trivial ones decay
     * 
     * @param llm_callback Function to call LLM for pattern extraction
     */
    void ConsolidateMemories(
        std::function<std::string(const std::string&)> llm_callback
    );
    
    // === Temporal Decay ===
    
    /**
     * Calculate memory strength based on time elapsed
     * Uses Ebbinghaus forgetting curve with emotional modulation
     * 
     * Formula: strength = base_strength * e^(-decay_rate * time) * emotional_boost
     * 
     * @param memory The episodic memory
     * @param current_time Current timestamp
     * @return Strength value (0 to 1)
     */
    float CalculateMemoryStrength(
        const EpisodicMemory& memory,
        int64_t current_time
    ) const;
    
    /**
     * Update all memory strengths based on current time
     */
    void UpdateMemoryStrengths();
    
    // === Persistence ===
    
    bool Save(const std::string& filepath);
    bool Load(const std::string& filepath);
    
    // === Statistics ===
    
    struct MemoryStats {
        int total_episodes;
        int active_episodes;  // strength > 0.1
        int total_semantic;
        float avg_episode_strength;
        int64_t oldest_memory;
        int64_t newest_memory;
    };
    
    MemoryStats GetStats() const;
    
    // === Configuration ===
    
    void SetDecayRate(float rate) { decay_rate_ = rate; }
    void SetEmotionalBoost(float boost) { emotional_boost_factor_ = boost; }
    void SetRetrievalBoost(float boost) { retrieval_boost_ = boost; }
    
private:
    std::vector<EpisodicMemory> episodic_memories_;
    std::vector<SemanticMemory> semantic_memories_;
    
    // Decay parameters (tunable)
    float decay_rate_ = 0.0001f;           // How fast memories fade (per second)
    float emotional_boost_factor_ = 2.0f;  // Multiplier for emotional memories
    float retrieval_boost_ = 0.1f;         // Strength increase per retrieval
    mutable std::mutex mutex_;
    
    // Helper functions
    int64_t GetCurrentTimestamp() const;
    float CalculateImportance(const std::string& description) const;
    std::string GenerateMemoryId() const;
    float CalculateSimilarity(const std::string& a, const std::string& b) const;
    
    // Consolidation helpers
    std::vector<std::vector<int>> ClusterEpisodes() const;
    std::string ExtractPattern(const std::vector<EpisodicMemory>& cluster,
                               std::function<std::string(const std::string&)> llm) const;
};

} // namespace NPCInference
