#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <mutex>
#include <deque>
#include <chrono>
#include <nlohmann/json.hpp>

namespace NPCInference {

/**
 * Ambient Awareness System
 * 
 * Features:
 * - Event Propagation: Information spreads through NPC network
 * - Inference Engine: NPCs deduce events from indirect evidence
 * - Plausibility Scoring: Assess likelihood of inferred events
 * - Source Tracking: Track information provenance and reliability
 * - Temporal Reasoning: Infer event timing from evidence
 */

struct ObservedEvent {
    std::string event_id;
    std::string event_type;       // e.g., "combat", "theft", "arrival"
    std::string description;
    std::vector<std::string> involved_entities;
    std::string location;
    int64_t timestamp;
    bool directly_witnessed;      // Did this NPC see it directly?
    float certainty;              // How certain are we? (0-1)
    std::vector<std::string> evidence_ids;  // What evidence supports this?
    
    ObservedEvent() : timestamp(0), directly_witnessed(false), certainty(1.0f) {}
};

struct Evidence {
    std::string evidence_id;
    std::string evidence_type;    // e.g., "visual", "auditory", "testimony", "physical"
    std::string description;
    std::string location;
    int64_t observed_at;
    float reliability;            // How reliable is this evidence? (0-1)
    std::vector<std::string> possible_causes;  // What events could cause this?
    
    Evidence() : observed_at(0), reliability(0.8f) {}
};

struct InferredEvent {
    std::string event_id;
    std::string event_type;
    std::string description;
    int64_t estimated_time;       // When we think it happened
    float plausibility;           // How plausible is this inference? (0-1)
    std::vector<std::string> supporting_evidence;
    std::string inference_method; // How did we infer this?
    bool confirmed;               // Has this been confirmed by direct observation?
    
    InferredEvent() : estimated_time(0), plausibility(0.5f), confirmed(false) {}
};

struct InformationSource {
    std::string source_id;        // Who/what provided this information
    float credibility;            // How credible is this source? (0-1)
    int correct_predictions;      // Track record
    int total_predictions;
    std::vector<std::string> known_biases;  // What biases does this source have?
    
    InformationSource() : credibility(0.7f), correct_predictions(0), total_predictions(0) {}
};

class AmbientAwarenessSystem {
public:
    AmbientAwarenessSystem();
    ~AmbientAwarenessSystem() = default;
    
    // === Event Observation ===
    
    /**
     * Record a directly observed event
     * @param event_type Type of event
     * @param description What happened
     * @param involved_entities Who was involved
     * @param location Where it happened
     * @return Event ID
     */
    std::string ObserveEvent(
        const std::string& event_type,
        const std::string& description,
        const std::vector<std::string>& involved_entities,
        const std::string& location
    );
    
    /**
     * Record indirect evidence
     * @param evidence_type Type of evidence (visual, auditory, etc.)
     * @param description What was observed
     * @param location Where
     * @param reliability How reliable (0-1)
     * @return Evidence ID
     */
    std::string RecordEvidence(
        const std::string& evidence_type,
        const std::string& description,
        const std::string& location,
        float reliability = 0.8f
    );
    
    // === Inference Engine ===
    
    /**
     * Analyze evidence and infer possible events
     * This is the core "reasoning" function
     */
    void InferEvents();
    
    /**
     * Get inferred events
     * @param min_plausibility Minimum plausibility threshold
     * @return Inferred events above threshold
     */
    std::vector<InferredEvent> GetInferences(float min_plausibility = 0.6f) const;
    
    /**
     * Check if NPC is aware of a specific event (directly or indirectly)
     * @param event_type Type of event to check
     * @return Confidence level (0-1)
     */
    float IsAwareOf(const std::string& event_type) const;
    
    // === Information Sharing ===
    
    /**
     * Receive information from another NPC
     * @param source_npc Who told us
     * @param event_description What they said
     * @param their_certainty How certain they were
     */
    void ReceiveInformation(
        const std::string& source_npc,
        const std::string& event_description,
        float their_certainty
    );
    
    /**
     * Share knowledge with another NPC
     * @param target_npc Who to tell
     * @return Information package to send
     */
    nlohmann::json ShareKnowledge(const std::string& target_npc) const;
    
    // === Source Credibility ===
    
    /**
     * Update source credibility based on new information
     * @param source_id Source to update
     * @param was_correct Was their information accurate?
     */
    void UpdateSourceCredibility(const std::string& source_id, bool was_correct);
    
    /**
     * Get credibility of a source
     * @param source_id Source to query
     * @return Credibility (0-1)
     */
    float GetSourceCredibility(const std::string& source_id) const;
    
    // === Temporal Reasoning ===
    
    /**
     * Estimate when an event likely occurred based on evidence
     * @param evidence_ids Evidence to analyze
     * @return Estimated timestamp
     */
    int64_t EstimateEventTime(const std::vector<std::string>& evidence_ids) const;
    
    /**
     * Check if two events are causally related
     * @param event_a First event
     * @param event_b Second event
     * @return Probability of causal relationship (0-1)
     */
    float AssessCausalRelationship(
        const std::string& event_a,
        const std::string& event_b
    ) const;
    
    // === Query Interface ===
    
    /**
     * Get all known events (direct + inferred)
     * @param min_certainty Minimum certainty threshold
     * @return All events above threshold
     */
    std::vector<ObservedEvent> GetAllKnownEvents(float min_certainty = 0.5f) const;
    
    /**
     * Get events involving a specific entity
     * @param entity_name Entity to search for
     * @return Events involving this entity
     */
    std::vector<ObservedEvent> GetEventsInvolving(const std::string& entity_name) const;
    
    /**
     * Get events at a specific location
     * @param location Location to search
     * @return Events at this location
     */
    std::vector<ObservedEvent> GetEventsAt(const std::string& location) const;
    
    // === Persistence ===
    
    bool Save(const std::string& filepath);
    bool Load(const std::string& filepath);
    nlohmann::json ToJSON() const;
    void FromJSON(const nlohmann::json& j);
    
    // === Statistics ===
    
    struct AwarenessStats {
        int direct_observations;
        int inferred_events;
        int evidence_pieces;
        float avg_inference_plausibility;
        int confirmed_inferences;
        float inference_accuracy;  // confirmed / total inferred
    };
    
    AwarenessStats GetStats() const;
    
private:
    std::vector<ObservedEvent> observed_events_;
    std::vector<Evidence> evidence_collection_;
    std::vector<InferredEvent> inferred_events_;
    std::map<std::string, InformationSource> information_sources_;
    
    // Inference rules
    struct InferenceRule {
        std::string rule_name;
        std::vector<std::string> required_evidence_types;
        std::string inferred_event_type;
        float base_plausibility;
        std::function<bool(const std::vector<Evidence>&)> condition;
    };
    std::vector<InferenceRule> inference_rules_;
    
    // Helper functions
    int64_t GetCurrentTimestamp() const;
    std::string GenerateEventId() const;
    std::string GenerateEvidenceId() const;
    void InitializeInferenceRules();
    float CalculateEvidenceStrength(const std::vector<std::string>& evidence_ids) const;
    std::vector<Evidence> GetEvidenceByIds(const std::vector<std::string>& ids) const;
    bool EvidenceSupportsEvent(const Evidence& evidence, const std::string& event_type) const;
    
    mutable std::mutex mutex_;
};

} // namespace NPCInference
