#pragma once

#include <string>
#include <vector>
#include <map>
#include <deque>
#include <mutex>
#include <chrono>
#include <nlohmann/json.hpp>

namespace NPCInference {

/**
 * Player Behavior Modeling System
 * 
 * Features:
 * - Action Tracking: Records player actions with context
 * - Pattern Detection: Identifies recurring strategies and preferences
 * - Predictive Modeling: Anticipates player's next move
 * - Adaptive Responses: NPCs adjust tactics based on player behavior
 * - Skill Assessment: Estimates player skill level dynamically
 */

struct PlayerAction {
    std::string action_type;      // e.g., "attack", "defend", "negotiate", "explore"
    std::string target;           // What/who the action targeted
    std::string context;          // Situation when action occurred
    int64_t timestamp;
    bool was_successful;          // Did the action achieve its goal?
    float risk_level;             // How risky was this action (0-1)
    std::map<std::string, float> parameters;  // Additional action details
    
    PlayerAction() : timestamp(0), was_successful(false), risk_level(0.5f) {}
};

struct BehaviorPattern {
    std::string pattern_id;
    std::string pattern_type;     // e.g., "aggressive", "cautious", "diplomatic"
    std::string description;      // Human-readable pattern description
    float confidence;             // How confident we are in this pattern (0-1)
    int occurrence_count;         // How many times we've seen this
    std::vector<std::string> supporting_actions;  // Action IDs that support this
    int64_t first_detected;
    int64_t last_seen;
    
    BehaviorPattern() : confidence(0.5f), occurrence_count(0), 
                       first_detected(0), last_seen(0) {}
};

struct PlayerProfile {
    // === Playstyle Dimensions ===
    float aggression;             // 0 (passive) to 1 (aggressive)
    float caution;                // 0 (reckless) to 1 (cautious)
    float social_preference;      // 0 (combat-focused) to 1 (dialogue-focused)
    float exploration_tendency;   // 0 (goal-oriented) to 1 (exploratory)
    float creativity;             // 0 (predictable) to 1 (creative/unpredictable)
    
    // === Skill Assessment ===
    float estimated_skill;        // 0 (novice) to 1 (expert)
    float reaction_speed;         // Based on action timing
    float strategic_thinking;     // Based on action sequencing
    
    // === Preferences ===
    std::map<std::string, float> preferred_actions;  // Action type -> frequency
    std::map<std::string, float> avoided_actions;    // Actions player rarely uses
    std::vector<std::string> dominant_patterns;      // Most common behavior patterns
    
    PlayerProfile() : aggression(0.5f), caution(0.5f), social_preference(0.5f),
                     exploration_tendency(0.5f), creativity(0.5f),
                     estimated_skill(0.5f), reaction_speed(0.5f), 
                     strategic_thinking(0.5f) {}
};

class PlayerBehaviorModeling {
public:
    PlayerBehaviorModeling();
    ~PlayerBehaviorModeling() = default;
    
    // === Action Tracking ===
    
    /**
     * Record a player action
     * @param action_type Type of action (attack, defend, etc.)
     * @param target What the action targeted
     * @param context Situation context
     * @param was_successful Did it work?
     * @param risk_level How risky (0-1)
     * @return Action ID
     */
    std::string RecordAction(
        const std::string& action_type,
        const std::string& target,
        const std::string& context,
        bool was_successful,
        float risk_level = 0.5f
    );
    
    /**
     * Get recent actions
     * @param count How many recent actions
     * @return Vector of recent actions
     */
    std::vector<PlayerAction> GetRecentActions(int count = 10) const;
    
    // === Pattern Detection ===
    
    /**
     * Analyze action history and detect behavioral patterns
     * This should be called periodically (e.g., every 10 actions)
     */
    void DetectPatterns();
    
    /**
     * Get detected patterns
     * @param min_confidence Minimum confidence threshold
     * @return Patterns above threshold
     */
    std::vector<BehaviorPattern> GetPatterns(float min_confidence = 0.6f) const;
    
    /**
     * Check if player exhibits a specific pattern
     * @param pattern_type Pattern to check for
     * @return Confidence level (0-1)
     */
    float HasPattern(const std::string& pattern_type) const;
    
    // === Player Profile ===
    
    /**
     * Get current player profile
     * Profile is updated automatically as actions are recorded
     */
    PlayerProfile GetProfile() const { return profile_; }
    
    /**
     * Update profile based on recent actions
     * Called automatically by RecordAction, but can be called manually
     */
    void UpdateProfile();
    
    // === Prediction ===
    
    /**
     * Predict player's next likely action
     * @param current_context Current situation
     * @param top_n Number of predictions to return
     * @return Predicted actions with confidence scores
     */
    std::vector<std::pair<std::string, float>> PredictNextAction(
        const std::string& current_context,
        int top_n = 3
    ) const;
    
    /**
     * Estimate probability player will choose a specific action
     * @param action_type Action to evaluate
     * @param context Current context
     * @return Probability (0-1)
     */
    float EstimateActionProbability(
        const std::string& action_type,
        const std::string& context
    ) const;
    
    // === Adaptive Response ===
    
    /**
     * Suggest NPC counter-strategy based on player behavior
     * @param npc_goal What the NPC is trying to achieve
     * @return Recommended strategy
     */
    std::string SuggestCounterStrategy(const std::string& npc_goal) const;
    
    /**
     * Assess how predictable the player is
     * @return Predictability score (0 = very unpredictable, 1 = very predictable)
     */
    float AssessPredictability() const;
    
    // === Persistence ===
    
    bool Save(const std::string& filepath);
    bool Load(const std::string& filepath);
    nlohmann::json ToJSON() const;
    void FromJSON(const nlohmann::json& j);
    
    // === Statistics ===
    
    struct ModelingStats {
        int total_actions;
        int patterns_detected;
        float avg_success_rate;
        float profile_confidence;
        std::string dominant_playstyle;
    };
    
    ModelingStats GetStats() const;
    
private:
    std::deque<PlayerAction> action_history_;  // Recent actions (limited size)
    std::vector<BehaviorPattern> detected_patterns_;
    PlayerProfile profile_;
    
    // Configuration
    int max_history_size_ = 100;  // Keep last 100 actions
    int pattern_detection_window_ = 20;  // Analyze last 20 actions for patterns
    
    // Helper functions
    int64_t GetCurrentTimestamp() const;
    std::string GenerateActionId() const;
    float CalculateActionSimilarity(const PlayerAction& a, const PlayerAction& b) const;
    std::vector<std::vector<int>> ClusterActions() const;
    BehaviorPattern AnalyzeCluster(const std::vector<PlayerAction>& cluster) const;
    void UpdatePlaystyleDimensions();
    void UpdateSkillAssessment();
    float CalculateEntropy() const;  // For predictability assessment
    
    mutable std::mutex mutex_;
};

} // namespace NPCInference
