#include "EmotionalContinuitySystem.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <sstream>

namespace NPCInference {

EmotionalContinuitySystem::EmotionalContinuitySystem() {
    // Default neutral personality
    personality_ = PersonalityProfile();
    
    // Default baseline: slightly positive
    baseline_mood_.joy = 0.6f;
    baseline_mood_.trust = 0.6f;
    baseline_mood_.anticipation = 0.5f;
    
    current_emotion_ = baseline_mood_;
}

EmotionalContinuitySystem::EmotionalContinuitySystem(const PersonalityProfile& personality)
    : personality_(personality) {
    // Baseline mood influenced by personality
    baseline_mood_.joy = 0.3f + (personality.agreeableness * 0.3f) + ((1.0f - personality.neuroticism) * 0.2f);
    baseline_mood_.trust = 0.3f + (personality.agreeableness * 0.4f);
    baseline_mood_.fear = personality.neuroticism * 0.3f;
    baseline_mood_.anticipation = 0.3f + (personality.openness * 0.3f);
    
    current_emotion_ = baseline_mood_;
}

void EmotionalContinuitySystem::SetPersonality(const PersonalityProfile& personality) {
    personality_ = personality;
    
    // Adjust baseline mood based on personality
    baseline_mood_.joy = 0.3f + (personality.agreeableness * 0.3f) + ((1.0f - personality.neuroticism) * 0.2f);
    baseline_mood_.trust = 0.3f + (personality.agreeableness * 0.4f);
    baseline_mood_.fear = personality.neuroticism * 0.3f;
    baseline_mood_.anticipation = 0.3f + (personality.openness * 0.3f);
}

PersonalityProfile EmotionalContinuitySystem::CreatePersonality(const std::string& archetype) {
    if (archetype == "cheerful") return Personalities::Cheerful();
    if (archetype == "grumpy") return Personalities::Grumpy();
    if (archetype == "cautious") return Personalities::Cautious();
    if (archetype == "bold") return Personalities::Bold();
    return PersonalityProfile();  // Default neutral
}

void EmotionalContinuitySystem::ApplyEmotionalStimulus(
    const EmotionalState& stimulus,
    float intensity,
    float inertia_override
) {
    // Calculate inertia (resistance to change)
    float inertia = (inertia_override >= 0.0f) ? inertia_override : CalculateInertia();
    
    // Blend factor: how much of the stimulus to apply
    float blend_factor = (1.0f - inertia) * intensity;
    
    // Blend current emotion with stimulus
    current_emotion_ = BlendEmotions(current_emotion_, stimulus, blend_factor);
    
    // Add to history
    emotion_history_.push_back(current_emotion_);
    if (emotion_history_.size() > MAX_HISTORY) {
        emotion_history_.erase(emotion_history_.begin());
    }
}

void EmotionalContinuitySystem::DecayTowardBaseline(float delta_time) {
    // Emotions gradually return to baseline
    float decay_factor = decay_rate_ * delta_time;
    current_emotion_ = BlendEmotions(current_emotion_, baseline_mood_, decay_factor);
}

void EmotionalContinuitySystem::SetBaselineMood(const EmotionalState& mood) {
    baseline_mood_ = mood;
}

void EmotionalContinuitySystem::UpdateSentiment(
    const std::string& entity_id,
    float sentiment_delta,
    float intensity_delta,
    const std::string& event_id
) {
    EntitySentiment& sentiment = sentiments_[entity_id];
    sentiment.entity_id = entity_id;
    
    // Update sentiment with inertia
    float current_sentiment = sentiment.sentiment;
    float target_sentiment = std::clamp(current_sentiment + sentiment_delta, -1.0f, 1.0f);
    
    // Emotional inertia applies to sentiment changes too
    float inertia = CalculateInertia();
    sentiment.sentiment = current_sentiment * inertia + target_sentiment * (1.0f - inertia);
    
    // Update intensity
    sentiment.intensity = std::clamp(sentiment.intensity + intensity_delta, 0.0f, 1.0f);
    
    sentiment.last_updated = GetCurrentTimestamp();
    
    if (!event_id.empty()) {
        sentiment.contributing_events.push_back(event_id);
    }
}

EntitySentiment EmotionalContinuitySystem::GetSentiment(const std::string& entity_id) const {
    auto it = sentiments_.find(entity_id);
    if (it != sentiments_.end()) {
        return it->second;
    }
    
    // Return neutral sentiment if not found
    EntitySentiment neutral;
    neutral.entity_id = entity_id;
    neutral.sentiment = 0.0f;
    neutral.intensity = 0.0f;
    return neutral;
}

std::vector<EntitySentiment> EmotionalContinuitySystem::GetAllSentiments() const {
    std::vector<EntitySentiment> results;
    for (const auto& [id, sentiment] : sentiments_) {
        results.push_back(sentiment);
    }
    return results;
}

EmotionalState EmotionalContinuitySystem::GenerateReaction(
    const std::string& event_type,
    float event_intensity,
    const std::string& involving_entity
) {
    // Get base emotional response for event type
    EmotionalState reaction = GetEmotionForEventType(event_type);
    
    // Modulate by personality
    if (event_type == "threat" || event_type == "danger") {
        reaction.fear *= (0.5f + personality_.neuroticism * 0.5f);
        reaction.anger *= (0.5f + (1.0f - personality_.agreeableness) * 0.5f);
    } else if (event_type == "gift" || event_type == "kindness") {
        reaction.joy *= (0.5f + personality_.agreeableness * 0.5f);
        reaction.trust *= (0.5f + personality_.agreeableness * 0.5f);
    } else if (event_type == "betrayal") {
        reaction.anger *= (0.5f + (1.0f - personality_.agreeableness) * 0.5f);
        reaction.sadness *= (0.5f + personality_.neuroticism * 0.5f);
    }
    
    // Consider existing sentiment toward entity
    if (!involving_entity.empty()) {
        auto sentiment = GetSentiment(involving_entity);
        
        // Positive sentiment amplifies positive events, dampens negative
        if (sentiment.sentiment > 0.0f) {
            if (event_type == "gift" || event_type == "help") {
                reaction.joy *= (1.0f + sentiment.sentiment * 0.5f);
            } else if (event_type == "betrayal" || event_type == "threat") {
                reaction.anger *= (1.0f + sentiment.sentiment * 0.5f);  // More hurt by betrayal from friend
                reaction.sadness *= (1.0f + sentiment.sentiment * 0.5f);
            }
        }
    }
    
    // Scale by event intensity
    reaction.joy *= event_intensity;
    reaction.trust *= event_intensity;
    reaction.fear *= event_intensity;
    reaction.surprise *= event_intensity;
    reaction.sadness *= event_intensity;
    reaction.disgust *= event_intensity;
    reaction.anger *= event_intensity;
    reaction.anticipation *= event_intensity;
    
    return reaction;
}

bool EmotionalContinuitySystem::WouldReactTo(
    const std::string& event_type,
    float event_intensity
) const {
    // Neurotic personalities react to weaker stimuli
    float threshold = reaction_threshold_ * (1.0f - personality_.neuroticism * 0.5f);
    return event_intensity >= threshold;
}

std::string EmotionalContinuitySystem::DescribeEmotion() const {
    std::string dominant = GetDominantEmotion();
    float intensity = GetEmotionalIntensity();
    
    std::ostringstream desc;
    
    if (intensity < 0.3f) {
        desc << "calm and " << dominant;
    } else if (intensity < 0.6f) {
        desc << "feeling " << dominant;
    } else {
        desc << "very " << dominant;
    }
    
    // Add valence descriptor
    float valence = current_emotion_.GetValence();
    if (valence > 0.3f) {
        desc << " (positive)";
    } else if (valence < -0.3f) {
        desc << " (negative)";
    }
    
    return desc.str();
}

std::string EmotionalContinuitySystem::GetDominantEmotion() const {
    struct EmotionScore {
        std::string name;
        float value;
    };
    
    std::vector<EmotionScore> emotions = {
        {"joyful", current_emotion_.joy},
        {"trusting", current_emotion_.trust},
        {"fearful", current_emotion_.fear},
        {"surprised", current_emotion_.surprise},
        {"sad", current_emotion_.sadness},
        {"disgusted", current_emotion_.disgust},
        {"angry", current_emotion_.anger},
        {"anticipating", current_emotion_.anticipation}
    };
    
    auto max_emotion = std::max_element(emotions.begin(), emotions.end(),
        [](const EmotionScore& a, const EmotionScore& b) {
            return a.value < b.value;
        });
    
    return max_emotion->name;
}

float EmotionalContinuitySystem::GetEmotionalIntensity() const {
    // Average of all emotion magnitudes
    float total = current_emotion_.joy + current_emotion_.trust + 
                  current_emotion_.fear + current_emotion_.surprise +
                  current_emotion_.sadness + current_emotion_.disgust +
                  current_emotion_.anger + current_emotion_.anticipation;
    return total / 8.0f;
}

bool EmotionalContinuitySystem::Save(const std::string& filepath) {
    nlohmann::json j;
    
    // Save personality
    j["personality"]["openness"] = personality_.openness;
    j["personality"]["conscientiousness"] = personality_.conscientiousness;
    j["personality"]["extraversion"] = personality_.extraversion;
    j["personality"]["agreeableness"] = personality_.agreeableness;
    j["personality"]["neuroticism"] = personality_.neuroticism;
    
    // Save current emotion
    j["current_emotion"]["joy"] = current_emotion_.joy;
    j["current_emotion"]["trust"] = current_emotion_.trust;
    j["current_emotion"]["fear"] = current_emotion_.fear;
    j["current_emotion"]["surprise"] = current_emotion_.surprise;
    j["current_emotion"]["sadness"] = current_emotion_.sadness;
    j["current_emotion"]["disgust"] = current_emotion_.disgust;
    j["current_emotion"]["anger"] = current_emotion_.anger;
    j["current_emotion"]["anticipation"] = current_emotion_.anticipation;
    
    // Save baseline mood
    j["baseline_mood"]["joy"] = baseline_mood_.joy;
    j["baseline_mood"]["trust"] = baseline_mood_.trust;
    j["baseline_mood"]["fear"] = baseline_mood_.fear;
    j["baseline_mood"]["surprise"] = baseline_mood_.surprise;
    j["baseline_mood"]["sadness"] = baseline_mood_.sadness;
    j["baseline_mood"]["disgust"] = baseline_mood_.disgust;
    j["baseline_mood"]["anger"] = baseline_mood_.anger;
    j["baseline_mood"]["anticipation"] = baseline_mood_.anticipation;
    
    // Save sentiments
    nlohmann::json sentiments = nlohmann::json::array();
    for (const auto& [id, sentiment] : sentiments_) {
        nlohmann::json s_json;
        s_json["entity_id"] = sentiment.entity_id;
        s_json["sentiment"] = sentiment.sentiment;
        s_json["intensity"] = sentiment.intensity;
        s_json["last_updated"] = sentiment.last_updated;
        s_json["events"] = sentiment.contributing_events;
        sentiments.push_back(s_json);
    }
    j["sentiments"] = sentiments;
    
    // Save config
    j["config"]["emotional_inertia"] = emotional_inertia_;
    j["config"]["decay_rate"] = decay_rate_;
    j["config"]["reaction_threshold"] = reaction_threshold_;
    
    std::ofstream file(filepath);
    if (!file.is_open()) return false;
    
    file << std::setw(2) << j << std::endl;
    return true;
}

bool EmotionalContinuitySystem::Load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;
    
    nlohmann::json j;
    try {
        file >> j;
    } catch (...) {
        return false;
    }
    
    // Load personality
    if (j.contains("personality")) {
        personality_.openness = j["personality"].value("openness", 0.5f);
        personality_.conscientiousness = j["personality"].value("conscientiousness", 0.5f);
        personality_.extraversion = j["personality"].value("extraversion", 0.5f);
        personality_.agreeableness = j["personality"].value("agreeableness", 0.5f);
        personality_.neuroticism = j["personality"].value("neuroticism", 0.5f);
    }
    
    // Load current emotion
    if (j.contains("current_emotion")) {
        current_emotion_.joy = j["current_emotion"].value("joy", 0.5f);
        current_emotion_.trust = j["current_emotion"].value("trust", 0.5f);
        current_emotion_.fear = j["current_emotion"].value("fear", 0.0f);
        current_emotion_.surprise = j["current_emotion"].value("surprise", 0.0f);
        current_emotion_.sadness = j["current_emotion"].value("sadness", 0.0f);
        current_emotion_.disgust = j["current_emotion"].value("disgust", 0.0f);
        current_emotion_.anger = j["current_emotion"].value("anger", 0.0f);
        current_emotion_.anticipation = j["current_emotion"].value("anticipation", 0.5f);
    }
    
    // Load baseline mood
    if (j.contains("baseline_mood")) {
        baseline_mood_.joy = j["baseline_mood"].value("joy", 0.5f);
        baseline_mood_.trust = j["baseline_mood"].value("trust", 0.5f);
        baseline_mood_.fear = j["baseline_mood"].value("fear", 0.0f);
        baseline_mood_.surprise = j["baseline_mood"].value("surprise", 0.0f);
        baseline_mood_.sadness = j["baseline_mood"].value("sadness", 0.0f);
        baseline_mood_.disgust = j["baseline_mood"].value("disgust", 0.0f);
        baseline_mood_.anger = j["baseline_mood"].value("anger", 0.0f);
        baseline_mood_.anticipation = j["baseline_mood"].value("anticipation", 0.5f);
    }
    
    // Load sentiments
    sentiments_.clear();
    if (j.contains("sentiments")) {
        for (const auto& s_json : j["sentiments"]) {
            EntitySentiment sentiment;
            sentiment.entity_id = s_json.value("entity_id", "");
            sentiment.sentiment = s_json.value("sentiment", 0.0f);
            sentiment.intensity = s_json.value("intensity", 0.0f);
            sentiment.last_updated = s_json.value("last_updated", 0L);
            sentiment.contributing_events = s_json.value("events", std::vector<std::string>{});
            sentiments_[sentiment.entity_id] = sentiment;
        }
    }
    
    // Load config
    if (j.contains("config")) {
        emotional_inertia_ = j["config"].value("emotional_inertia", 0.7f);
        decay_rate_ = j["config"].value("decay_rate", 0.1f);
        reaction_threshold_ = j["config"].value("reaction_threshold", 0.3f);
    }
    
    return true;
}

EmotionalContinuitySystem::EmotionalStats EmotionalContinuitySystem::GetStats() const {
    EmotionalStats stats;
    
    // Calculate average valence and arousal
    if (!emotion_history_.empty()) {
        float total_valence = 0.0f;
        float total_arousal = 0.0f;
        
        for (const auto& emotion : emotion_history_) {
            total_valence += emotion.GetValence();
            total_arousal += emotion.GetArousal();
        }
        
        stats.avg_valence = total_valence / emotion_history_.size();
        stats.avg_arousal = total_arousal / emotion_history_.size();
        
        // Calculate volatility (standard deviation of valence)
        float variance = 0.0f;
        for (const auto& emotion : emotion_history_) {
            float diff = emotion.GetValence() - stats.avg_valence;
            variance += diff * diff;
        }
        stats.emotional_volatility = std::sqrt(variance / emotion_history_.size());
    } else {
        stats.avg_valence = current_emotion_.GetValence();
        stats.avg_arousal = current_emotion_.GetArousal();
        stats.emotional_volatility = 0.0f;
    }
    
    stats.num_sentiments = sentiments_.size();
    stats.num_strong_sentiments = 0;
    for (const auto& [id, sentiment] : sentiments_) {
        if (sentiment.intensity > 0.7f) {
            stats.num_strong_sentiments++;
        }
    }
    
    return stats;
}

// === Private Helper Functions ===

EmotionalState EmotionalContinuitySystem::BlendEmotions(
    const EmotionalState& current,
    const EmotionalState& target,
    float blend_factor
) const {
    EmotionalState result;
    
    blend_factor = std::clamp(blend_factor, 0.0f, 1.0f);
    float keep_factor = 1.0f - blend_factor;
    
    result.joy = current.joy * keep_factor + target.joy * blend_factor;
    result.trust = current.trust * keep_factor + target.trust * blend_factor;
    result.fear = current.fear * keep_factor + target.fear * blend_factor;
    result.surprise = current.surprise * keep_factor + target.surprise * blend_factor;
    result.sadness = current.sadness * keep_factor + target.sadness * blend_factor;
    result.disgust = current.disgust * keep_factor + target.disgust * blend_factor;
    result.anger = current.anger * keep_factor + target.anger * blend_factor;
    result.anticipation = current.anticipation * keep_factor + target.anticipation * blend_factor;
    
    return result;
}

float EmotionalContinuitySystem::CalculateInertia() const {
    // Base inertia from configuration
    float inertia = emotional_inertia_;
    
    // Neurotic personalities have less inertia (more volatile)
    inertia *= (1.0f - personality_.neuroticism * 0.3f);
    
    // Conscientious personalities have more inertia (more stable)
    inertia *= (1.0f + personality_.conscientiousness * 0.2f);
    
    return std::clamp(inertia, 0.0f, 0.95f);
}

int64_t EmotionalContinuitySystem::GetCurrentTimestamp() const {
    return std::chrono::system_clock::now().time_since_epoch().count() / 1000000000;
}

EmotionalState EmotionalContinuitySystem::GetEmotionForEventType(const std::string& event_type) const {
    EmotionalState emotion;
    
    if (event_type == "gift" || event_type == "kindness") {
        emotion.joy = 0.8f;
        emotion.trust = 0.6f;
        emotion.anticipation = 0.4f;
    } else if (event_type == "betrayal") {
        emotion.anger = 0.8f;
        emotion.sadness = 0.7f;
        emotion.disgust = 0.5f;
    } else if (event_type == "threat" || event_type == "danger") {
        emotion.fear = 0.9f;
        emotion.surprise = 0.5f;
        emotion.anticipation = 0.6f;
    } else if (event_type == "loss") {
        emotion.sadness = 0.9f;
        emotion.anger = 0.3f;
    } else if (event_type == "victory" || event_type == "success") {
        emotion.joy = 0.9f;
        emotion.trust = 0.5f;
        emotion.anticipation = 0.6f;
    } else if (event_type == "insult") {
        emotion.anger = 0.7f;
        emotion.disgust = 0.5f;
    } else if (event_type == "surprise_positive") {
        emotion.surprise = 0.8f;
        emotion.joy = 0.6f;
    } else if (event_type == "surprise_negative") {
        emotion.surprise = 0.8f;
        emotion.fear = 0.6f;
    }
    
    return emotion;
}

} // namespace NPCInference
