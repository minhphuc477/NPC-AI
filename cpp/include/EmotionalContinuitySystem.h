#pragma once

#include <string>
#include <map>
#include <vector>
#include <mutex>
#include <nlohmann/json.hpp>

namespace NPCInference {

/**
 * Emotional Continuity System - Persistent NPC Emotions
 * 
 * Features:
 * - Multi-Dimensional Emotions: Based on Plutchik's wheel of emotions
 * - Personality Profiles: Big Five personality traits
 * - Emotional Inertia: Emotions change gradually, not instantly
 * - Entity Sentiment: Track feelings toward specific characters
 * - Personality-Driven Reactions: Same event triggers different emotions
 * - Emotional memory (events trigger emotional responses)
 * - Mood system (baseline emotional state)
 */

// Big Five personality traits
struct PersonalityProfile {
    float openness = 0.5f;          // 0 (closed) to 1 (open)
    float conscientiousness = 0.5f; // 0 (careless) to 1 (careful)
    float extraversion = 0.5f;      // 0 (introvert) to 1 (extravert)
    float agreeableness = 0.5f;     // 0 (hostile) to 1 (friendly)
    float neuroticism = 0.5f;       // 0 (stable) to 1 (neurotic)
    
    // Derived traits
    float GetEmotionalStability() const { return 1.0f - neuroticism; }
    float GetSociability() const { return extraversion; }
};

// Multi-dimensional emotional state (Plutchik's wheel of emotions)
struct EmotionalState {
    // Primary emotions (0 to 1)
    float joy = 0.5f;
    float trust = 0.5f;
    float fear = 0.0f;
    float surprise = 0.0f;
    float sadness = 0.0f;
    float disgust = 0.0f;
    float anger = 0.0f;
    float anticipation = 0.5f;
    
    // Derived metrics
    float GetValence() const {
        return (joy + trust - fear - sadness - disgust - anger) / 6.0f;
    }
    
    float GetArousal() const {
        return (joy + fear + surprise + anger + anticipation) / 5.0f;
    }
    
    float GetDominance() const {
        return (trust + anger + anticipation - fear - sadness) / 5.0f;
    }
};

// Sentiment toward a specific entity
struct EntitySentiment {
    std::string entity_id;
    float sentiment = 0.0f;  // -1 (hate) to 1 (love)
    float intensity = 0.0f;  // 0 (indifferent) to 1 (passionate)
    int64_t last_updated = 0;
    std::vector<std::string> contributing_events;  // Event IDs that shaped this
};

class EmotionalContinuitySystem {
public:
    EmotionalContinuitySystem();
    explicit EmotionalContinuitySystem(const PersonalityProfile& personality);
    ~EmotionalContinuitySystem() = default;
    
    // === Personality Management ===
    
    void SetPersonality(const PersonalityProfile& personality);
    PersonalityProfile GetPersonality() const { return personality_; }
    
    /**
     * Create preset personalities
     */
    static PersonalityProfile CreatePersonality(const std::string& archetype);
    
    // === Emotional State Management ===
    
    /**
     * Get current emotional state
     */
    EmotionalState GetCurrentEmotion() const { return current_emotion_; }
    
    /**
     * Apply emotional stimulus (with inertia)
     * Emotions don't change instantly - they blend with current state
     * 
     * @param stimulus The target emotional state
     * @param intensity How strong the stimulus is (0 to 1)
     * @param inertia_override Optional override for inertia (default uses personality)
     */
    void ApplyEmotionalStimulus(
        const EmotionalState& stimulus,
        float intensity = 1.0f,
        float inertia_override = -1.0f
    );
    
    /**
     * Emotional decay toward baseline mood
     * Call this periodically to let emotions return to normal
     */
    void DecayTowardBaseline(float delta_time);
    
    /**
     * Set baseline mood (default emotional state)
     */
    void SetBaselineMood(const EmotionalState& mood);
    EmotionalState GetBaselineMood() const { return baseline_mood_; }
    
    // === Entity Sentiment ===
    
    /**
     * Update sentiment toward an entity
     * @param entity_id Who/what the sentiment is about
     * @param sentiment_delta Change in sentiment (-1 to 1)
     * @param intensity_delta Change in intensity (0 to 1)
     * @param event_id Optional event that caused this
     */
    void UpdateSentiment(
        const std::string& entity_id,
        float sentiment_delta,
        float intensity_delta = 0.0f,
        const std::string& event_id = ""
    );
    
    /**
     * Get sentiment toward an entity
     */
    EntitySentiment GetSentiment(const std::string& entity_id) const;
    
    /**
     * Get all sentiments
     */
    std::vector<EntitySentiment> GetAllSentiments() const;
    
    // === Emotional Reactions ===
    
    /**
     * Generate emotional reaction to an event
     * Considers personality and current emotional state
     * 
     * @param event_type Type of event (e.g., "betrayal", "gift", "threat")
     * @param event_intensity How intense the event is (0 to 1)
     * @param involving_entity Optional entity involved in the event
     * @return Resulting emotional state
     */
    EmotionalState GenerateReaction(
        const std::string& event_type,
        float event_intensity,
        const std::string& involving_entity = ""
    );
    
    /**
     * Check if NPC would react emotionally to an event
     * @return true if reaction threshold is met
     */
    bool WouldReactTo(
        const std::string& event_type,
        float event_intensity
    ) const;
    
    // === Emotional Expression ===
    
    /**
     * Get text description of current emotional state
     */
    std::string DescribeEmotion() const;
    
    /**
     * Get dominant emotion
     */
    std::string GetDominantEmotion() const;
    
    /**
     * Get emotional intensity (how strongly they're feeling)
     */
    float GetEmotionalIntensity() const;
    
    // === Persistence ===
    
    bool Save(const std::string& filepath);
    bool Load(const std::string& filepath);
    
    // === Statistics ===
    
    struct EmotionalStats {
        float avg_valence;      // How positive/negative
        float avg_arousal;      // How energized
        float emotional_volatility;  // How much emotions fluctuate
        int num_sentiments;
        int num_strong_sentiments;  // intensity > 0.7
    };
    
    EmotionalStats GetStats() const;
    
    // === Configuration ===
    
    void SetEmotionalInertia(float inertia) { emotional_inertia_ = inertia; }
    void SetDecayRate(float rate) { decay_rate_ = rate; }
    void SetReactionThreshold(float threshold) { reaction_threshold_ = threshold; }
    
private:
    mutable std::mutex mutex_;
    PersonalityProfile personality_;
    EmotionalState current_emotion_;
    EmotionalState baseline_mood_;
    
    // Entity sentiments: entity_id -> sentiment
    std::map<std::string, EntitySentiment> sentiments_;
    
    // Emotional history (for volatility calculation)
    std::vector<EmotionalState> emotion_history_;
    static constexpr int MAX_HISTORY = 100;
    
    // Configuration
    float emotional_inertia_ = 0.7f;  // How resistant to change (0 = instant, 1 = no change)
    float decay_rate_ = 0.1f;         // How fast emotions return to baseline
    float reaction_threshold_ = 0.3f; // Minimum intensity to trigger reaction
    
    // Helper functions
    EmotionalState BlendEmotions(
        const EmotionalState& current,
        const EmotionalState& target,
        float blend_factor
    ) const;
    
    float CalculateInertia() const;
    int64_t GetCurrentTimestamp() const;
    
    // Event type to emotion mapping
    EmotionalState GetEmotionForEventType(const std::string& event_type) const;
};

// === Preset Personalities ===

namespace Personalities {
    inline PersonalityProfile Cheerful() {
        PersonalityProfile p;
        p.openness = 0.7f;
        p.conscientiousness = 0.6f;
        p.extraversion = 0.8f;
        p.agreeableness = 0.8f;
        p.neuroticism = 0.2f;
        return p;
    }
    
    inline PersonalityProfile Grumpy() {
        PersonalityProfile p;
        p.openness = 0.3f;
        p.conscientiousness = 0.7f;
        p.extraversion = 0.2f;
        p.agreeableness = 0.3f;
        p.neuroticism = 0.6f;
        return p;
    }
    
    inline PersonalityProfile Cautious() {
        PersonalityProfile p;
        p.openness = 0.4f;
        p.conscientiousness = 0.8f;
        p.extraversion = 0.3f;
        p.agreeableness = 0.5f;
        p.neuroticism = 0.7f;
        return p;
    }
    
    inline PersonalityProfile Bold() {
        PersonalityProfile p;
        p.openness = 0.8f;
        p.conscientiousness = 0.4f;
        p.extraversion = 0.9f;
        p.agreeableness = 0.5f;
        p.neuroticism = 0.2f;
        return p;
    }
}

} // namespace NPCInference
