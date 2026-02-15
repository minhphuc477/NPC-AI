#include "AmbientAwarenessSystem.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>

namespace NPCInference {

AmbientAwarenessSystem::AmbientAwarenessSystem() {
    InitializeInferenceRules();
}

std::string AmbientAwarenessSystem::ObserveEvent(
    const std::string& event_type,
    const std::string& description,
    const std::vector<std::string>& involved_entities,
    const std::string& location
) {
    ObservedEvent event;
    event.event_id = GenerateEventId();
    event.event_type = event_type;
    event.description = description;
    event.involved_entities = involved_entities;
    event.location = location;
    event.timestamp = GetCurrentTimestamp();
    event.directly_witnessed = true;
    event.certainty = 1.0f;  // Direct observation = 100% certain
    
    observed_events_.push_back(event);
    return event.event_id;
}

std::string AmbientAwarenessSystem::RecordEvidence(
    const std::string& evidence_type,
    const std::string& description,
    const std::string& location,
    float reliability
) {
    Evidence evidence;
    evidence.evidence_id = GenerateEvidenceId();
    evidence.evidence_type = evidence_type;
    evidence.description = description;
    evidence.location = location;
    evidence.observed_at = GetCurrentTimestamp();
    evidence.reliability = std::clamp(reliability, 0.0f, 1.0f);
    
    // Determine possible causes based on evidence type and description
    if (evidence_type == "auditory") {
        if (description.find("scream") != std::string::npos || 
            description.find("shout") != std::string::npos) {
            evidence.possible_causes.push_back("combat");
            evidence.possible_causes.push_back("distress");
        } else if (description.find("footsteps") != std::string::npos) {
            evidence.possible_causes.push_back("movement");
            evidence.possible_causes.push_back("arrival");
        }
    } else if (evidence_type == "visual") {
        if (description.find("blood") != std::string::npos) {
            evidence.possible_causes.push_back("combat");
            evidence.possible_causes.push_back("injury");
        } else if (description.find("smoke") != std::string::npos) {
            evidence.possible_causes.push_back("fire");
        }
    } else if (evidence_type == "physical") {
        if (description.find("broken") != std::string::npos || 
            description.find("damaged") != std::string::npos) {
            evidence.possible_causes.push_back("combat");
            evidence.possible_causes.push_back("vandalism");
        }
    }
    
    evidence_collection_.push_back(evidence);
    return evidence.evidence_id;
}

void AmbientAwarenessSystem::InferEvents() {
    // Clear old low-plausibility inferences
    inferred_events_.erase(
        std::remove_if(inferred_events_.begin(), inferred_events_.end(),
            [](const InferredEvent& e) { return e.plausibility < 0.3f && !e.confirmed; }),
        inferred_events_.end()
    );
    
    // === Rule-Based Inference ===
    
    // Rule 1: Multiple combat sounds + blood evidence = combat event
    std::vector<Evidence> combat_sounds;
    std::vector<Evidence> blood_evidence;
    
    for (const auto& evidence : evidence_collection_) {
        if (evidence.evidence_type == "auditory" && 
            (evidence.description.find("clash") != std::string::npos ||
             evidence.description.find("scream") != std::string::npos)) {
            combat_sounds.push_back(evidence);
        }
        if (evidence.evidence_type == "visual" && 
            evidence.description.find("blood") != std::string::npos) {
            blood_evidence.push_back(evidence);
        }
    }
    
    if (!combat_sounds.empty() && !blood_evidence.empty()) {
        InferredEvent inference;
        inference.event_id = GenerateEventId();
        inference.event_type = "combat";
        inference.description = "Inferred combat event from sounds and blood evidence";
        inference.estimated_time = combat_sounds[0].observed_at;
        
        // Calculate plausibility based on evidence strength
        float sound_strength = std::min(1.0f, combat_sounds.size() * 0.3f);
        float visual_strength = std::min(1.0f, blood_evidence.size() * 0.4f);
        inference.plausibility = (sound_strength + visual_strength) / 2.0f;
        
        for (const auto& e : combat_sounds) {
            inference.supporting_evidence.push_back(e.evidence_id);
        }
        for (const auto& e : blood_evidence) {
            inference.supporting_evidence.push_back(e.evidence_id);
        }
        
        inference.inference_method = "Multi-evidence correlation (auditory + visual)";
        inferred_events_.push_back(inference);
    }
    
    // Rule 2: Footsteps approaching + door sounds = arrival event
    std::vector<Evidence> footsteps;
    std::vector<Evidence> door_sounds;
    
    for (const auto& evidence : evidence_collection_) {
        if (evidence.evidence_type == "auditory" && 
            evidence.description.find("footsteps") != std::string::npos) {
            footsteps.push_back(evidence);
        }
        if (evidence.evidence_type == "auditory" && 
            (evidence.description.find("door") != std::string::npos ||
             evidence.description.find("knock") != std::string::npos)) {
            door_sounds.push_back(evidence);
        }
    }
    
    if (!footsteps.empty() && !door_sounds.empty()) {
        InferredEvent inference;
        inference.event_id = GenerateEventId();
        inference.event_type = "arrival";
        inference.description = "Inferred arrival from footsteps and door sounds";
        inference.estimated_time = door_sounds[0].observed_at;
        inference.plausibility = 0.8f;  // High confidence for this pattern
        
        for (const auto& e : footsteps) {
            inference.supporting_evidence.push_back(e.evidence_id);
        }
        for (const auto& e : door_sounds) {
            inference.supporting_evidence.push_back(e.evidence_id);
        }
        
        inference.inference_method = "Sequential pattern matching";
        inferred_events_.push_back(inference);
    }
    
    // Rule 3: Smoke + heat + crackling = fire event
    std::vector<Evidence> fire_indicators;
    
    for (const auto& evidence : evidence_collection_) {
        if (evidence.description.find("smoke") != std::string::npos ||
            evidence.description.find("heat") != std::string::npos ||
            evidence.description.find("crackling") != std::string::npos ||
            evidence.description.find("burning") != std::string::npos) {
            fire_indicators.push_back(evidence);
        }
    }
    
    if (fire_indicators.size() >= 2) {
        InferredEvent inference;
        inference.event_id = GenerateEventId();
        inference.event_type = "fire";
        inference.description = "Inferred fire event from multiple indicators";
        inference.estimated_time = fire_indicators[0].observed_at;
        inference.plausibility = std::min(1.0f, fire_indicators.size() * 0.4f);
        
        for (const auto& e : fire_indicators) {
            inference.supporting_evidence.push_back(e.evidence_id);
        }
        
        inference.inference_method = "Multi-sensory convergence";
        inferred_events_.push_back(inference);
    }
    
    // Rule 4: Broken items + missing valuables = theft event
    std::vector<Evidence> damage_evidence;
    std::vector<Evidence> missing_items;
    
    for (const auto& evidence : evidence_collection_) {
        if (evidence.description.find("broken") != std::string::npos ||
            evidence.description.find("forced") != std::string::npos) {
            damage_evidence.push_back(evidence);
        }
        if (evidence.description.find("missing") != std::string::npos ||
            evidence.description.find("gone") != std::string::npos) {
            missing_items.push_back(evidence);
        }
    }
    
    if (!damage_evidence.empty() && !missing_items.empty()) {
        InferredEvent inference;
        inference.event_id = GenerateEventId();
        inference.event_type = "theft";
        inference.description = "Inferred theft from forced entry and missing items";
        inference.estimated_time = damage_evidence[0].observed_at;
        inference.plausibility = 0.75f;
        
        for (const auto& e : damage_evidence) {
            inference.supporting_evidence.push_back(e.evidence_id);
        }
        for (const auto& e : missing_items) {
            inference.supporting_evidence.push_back(e.evidence_id);
        }
        
        inference.inference_method = "Causal reasoning";
        inferred_events_.push_back(inference);
    }
}

std::vector<InferredEvent> AmbientAwarenessSystem::GetInferences(float min_plausibility) const {
    std::vector<InferredEvent> filtered;
    for (const auto& inference : inferred_events_) {
        if (inference.plausibility >= min_plausibility) {
            filtered.push_back(inference);
        }
    }
    return filtered;
}

float AmbientAwarenessSystem::IsAwareOf(const std::string& event_type) const {
    float max_certainty = 0.0f;
    
    // Check direct observations
    for (const auto& event : observed_events_) {
        if (event.event_type == event_type) {
            max_certainty = std::max(max_certainty, event.certainty);
        }
    }
    
    // Check inferences
    for (const auto& inference : inferred_events_) {
        if (inference.event_type == event_type) {
            max_certainty = std::max(max_certainty, inference.plausibility);
        }
    }
    
    return max_certainty;
}

void AmbientAwarenessSystem::ReceiveInformation(
    const std::string& source_npc,
    const std::string& event_description,
    float their_certainty
) {
    // Get or create source credibility
    if (information_sources_.find(source_npc) == information_sources_.end()) {
        InformationSource source;
        source.source_id = source_npc;
        source.credibility = 0.7f;  // Default credibility
        information_sources_[source_npc] = source;
    }
    
    float source_credibility = information_sources_[source_npc].credibility;
    
    // Adjust certainty based on source credibility
    float adjusted_certainty = their_certainty * source_credibility;
    
    // Record as evidence
    Evidence testimony;
    testimony.evidence_id = GenerateEvidenceId();
    testimony.evidence_type = "testimony";
    testimony.description = event_description + " (from " + source_npc + ")";
    testimony.observed_at = GetCurrentTimestamp();
    testimony.reliability = adjusted_certainty;
    
    evidence_collection_.push_back(testimony);
    
    // Re-run inference with new information
    InferEvents();
}

nlohmann::json AmbientAwarenessSystem::ShareKnowledge(const std::string& target_npc) const {
    nlohmann::json knowledge;
    
    // Share high-confidence observations
    knowledge["observations"] = nlohmann::json::array();
    for (const auto& event : observed_events_) {
        if (event.certainty > 0.7f) {
            nlohmann::json obs;
            obs["type"] = event.event_type;
            obs["description"] = event.description;
            obs["certainty"] = event.certainty;
            obs["location"] = event.location;
            knowledge["observations"].push_back(obs);
        }
    }
    
    // Share plausible inferences
    knowledge["inferences"] = nlohmann::json::array();
    for (const auto& inference : inferred_events_) {
        if (inference.plausibility > 0.6f) {
            nlohmann::json inf;
            inf["type"] = inference.event_type;
            inf["description"] = inference.description;
            inf["plausibility"] = inference.plausibility;
            knowledge["inferences"].push_back(inf);
        }
    }
    
    return knowledge;
}

void AmbientAwarenessSystem::UpdateSourceCredibility(const std::string& source_id, bool was_correct) {
    if (information_sources_.find(source_id) == information_sources_.end()) {
        InformationSource source;
        source.source_id = source_id;
        information_sources_[source_id] = source;
    }
    
    auto& source = information_sources_[source_id];
    source.total_predictions++;
    if (was_correct) {
        source.correct_predictions++;
    }
    
    // Update credibility based on track record
    if (source.total_predictions > 0) {
        source.credibility = static_cast<float>(source.correct_predictions) / source.total_predictions;
    }
}

float AmbientAwarenessSystem::GetSourceCredibility(const std::string& source_id) const {
    auto it = information_sources_.find(source_id);
    if (it != information_sources_.end()) {
        return it->second.credibility;
    }
    return 0.5f;  // Default neutral credibility
}

int64_t AmbientAwarenessSystem::EstimateEventTime(const std::vector<std::string>& evidence_ids) const {
    if (evidence_ids.empty()) return GetCurrentTimestamp();
    
    auto evidence_list = GetEvidenceByIds(evidence_ids);
    if (evidence_list.empty()) return GetCurrentTimestamp();
    
    // Use earliest evidence timestamp as estimate
    int64_t earliest = evidence_list[0].observed_at;
    for (const auto& evidence : evidence_list) {
        earliest = std::min(earliest, evidence.observed_at);
    }
    
    return earliest;
}

std::vector<ObservedEvent> AmbientAwarenessSystem::GetAllKnownEvents(float min_certainty) const {
    std::vector<ObservedEvent> all_events = observed_events_;
    
    // Add inferred events as observed events (with lower certainty)
    for (const auto& inference : inferred_events_) {
        if (inference.plausibility >= min_certainty) {
            ObservedEvent event;
            event.event_id = inference.event_id;
            event.event_type = inference.event_type;
            event.description = inference.description + " (inferred)";
            event.timestamp = inference.estimated_time;
            event.directly_witnessed = false;
            event.certainty = inference.plausibility;
            event.evidence_ids = inference.supporting_evidence;
            all_events.push_back(event);
        }
    }
    
    return all_events;
}

std::vector<ObservedEvent> AmbientAwarenessSystem::GetEventsInvolving(const std::string& entity_name) const {
    std::vector<ObservedEvent> filtered;
    for (const auto& event : observed_events_) {
        if (std::find(event.involved_entities.begin(), event.involved_entities.end(), entity_name) 
            != event.involved_entities.end()) {
            filtered.push_back(event);
        }
    }
    return filtered;
}

std::vector<ObservedEvent> AmbientAwarenessSystem::GetEventsAt(const std::string& location) const {
    std::vector<ObservedEvent> filtered;
    for (const auto& event : observed_events_) {
        if (event.location == location) {
            filtered.push_back(event);
        }
    }
    return filtered;
}

AmbientAwarenessSystem::AwarenessStats AmbientAwarenessSystem::GetStats() const {
    AwarenessStats stats;
    stats.direct_observations = static_cast<int>(observed_events_.size());
    stats.inferred_events = static_cast<int>(inferred_events_.size());
    stats.evidence_pieces = static_cast<int>(evidence_collection_.size());
    
    if (!inferred_events_.empty()) {
        float total_plausibility = 0.0f;
        int confirmed = 0;
        for (const auto& inference : inferred_events_) {
            total_plausibility += inference.plausibility;
            if (inference.confirmed) confirmed++;
        }
        stats.avg_inference_plausibility = total_plausibility / inferred_events_.size();
        stats.confirmed_inferences = confirmed;
        stats.inference_accuracy = inferred_events_.size() > 0 ? 
            static_cast<float>(confirmed) / inferred_events_.size() : 0.0f;
    } else {
        stats.avg_inference_plausibility = 0.0f;
        stats.confirmed_inferences = 0;
        stats.inference_accuracy = 0.0f;
    }
    
    return stats;
}

bool AmbientAwarenessSystem::Save(const std::string& filepath) {
    try {
        nlohmann::json j = ToJSON();
        std::ofstream file(filepath);
        file << std::setw(2) << j;
        return true;
    } catch (...) {
        return false;
    }
}

bool AmbientAwarenessSystem::Load(const std::string& filepath) {
    try {
        std::ifstream file(filepath);
        nlohmann::json j;
        file >> j;
        FromJSON(j);
        return true;
    } catch (...) {
        return false;
    }
}

nlohmann::json AmbientAwarenessSystem::ToJSON() const {
    nlohmann::json j;
    
    // Save observations
    j["observations"] = nlohmann::json::array();
    for (const auto& event : observed_events_) {
        nlohmann::json e;
        e["type"] = event.event_type;
        e["description"] = event.description;
        e["location"] = event.location;
        e["certainty"] = event.certainty;
        e["direct"] = event.directly_witnessed;
        j["observations"].push_back(e);
    }
    
    // Save inferences
    j["inferences"] = nlohmann::json::array();
    for (const auto& inference : inferred_events_) {
        nlohmann::json i;
        i["type"] = inference.event_type;
        i["description"] = inference.description;
        i["plausibility"] = inference.plausibility;
        i["method"] = inference.inference_method;
        j["inferences"].push_back(i);
    }
    
    // Save evidence
    j["evidence"] = nlohmann::json::array();
    for (const auto& evidence : evidence_collection_) {
        nlohmann::json ev;
        ev["type"] = evidence.evidence_type;
        ev["description"] = evidence.description;
        ev["reliability"] = evidence.reliability;
        j["evidence"].push_back(ev);
    }
    
    return j;
}

void AmbientAwarenessSystem::FromJSON(const nlohmann::json& j) {
    // Load would go here - simplified for now
}

void AmbientAwarenessSystem::InitializeInferenceRules() {
    // Rules are implemented directly in InferEvents() for now
    // Could be externalized to a rule engine in the future
}

std::vector<Evidence> AmbientAwarenessSystem::GetEvidenceByIds(const std::vector<std::string>& ids) const {
    std::vector<Evidence> result;
    for (const auto& id : ids) {
        for (const auto& evidence : evidence_collection_) {
            if (evidence.evidence_id == id) {
                result.push_back(evidence);
                break;
            }
        }
    }
    return result;
}

int64_t AmbientAwarenessSystem::GetCurrentTimestamp() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

std::string AmbientAwarenessSystem::GenerateEventId() const {
    std::stringstream ss;
    ss << "event_" << GetCurrentTimestamp() << "_" << observed_events_.size();
    return ss.str();
}

std::string AmbientAwarenessSystem::GenerateEvidenceId() const {
    std::stringstream ss;
    ss << "evidence_" << GetCurrentTimestamp() << "_" << evidence_collection_.size();
    return ss.str();
}

} // namespace NPCInference
