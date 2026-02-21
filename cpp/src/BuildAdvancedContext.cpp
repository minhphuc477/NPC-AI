#include "NPCInference.h"
#include "TemporalMemorySystem.h"
#include "SocialFabricNetwork.h"
#include "EmotionalContinuitySystem.h"
#include "PlayerBehaviorModeling.h"
#include "AmbientAwarenessSystem.h"

namespace NPCInference {

nlohmann::json NPCInferenceEngine::BuildAdvancedContext(const std::string& npc_id, const std::string& query) {
    nlohmann::json context;
    
    if (!temporal_memory_ || !social_fabric_network_ || !emotional_continuity_system_) {
        return context;  // Return empty if systems not initialized
    }
    
    // === 1. Temporal Memory Context ===
    auto relevant_memories = temporal_memory_->RetrieveEpisodes(query, 5, 0.1f);
    nlohmann::json memories = nlohmann::json::array();
    for (const auto& memory : relevant_memories) {
        nlohmann::json mem;
        mem["content"] = memory.description;
        mem["timestamp"] = memory.timestamp;
        mem["importance"] = memory.importance;
        mem["emotional_arousal"] = memory.emotional_arousal;
        mem["strength"] = memory.current_strength;
        memories.push_back(mem);
    }
    context["memories"] = memories;
    
    // === 2. Social Fabric Context ===
    
    // Get relationships - Limit to top 5 strongest relationships
    auto relationships = social_fabric_network_->GetAllRelationships(npc_id);
    std::sort(relationships.begin(), relationships.end(), [](const auto& a, const auto& b) {
        return a.GetStrength() > b.GetStrength();
    });
    
    nlohmann::json rels = nlohmann::json::array();
    for (size_t i = 0; i < std::min<size_t>(relationships.size(), 5); ++i) {
        const auto& rel = relationships[i];
        nlohmann::json r;
        r["entity"] = rel.npc_b;
        r["trust"] = rel.trust;
        r["affection"] = rel.affection;
        r["respect"] = rel.respect;
        r["strength"] = rel.GetStrength();
        r["is_positive"] = rel.IsPositive();
        rels.push_back(r);
    }
    context["relationships"] = rels;
    
    // Get gossip heard - Limit to top 5 most credible gossip
    auto gossip = social_fabric_network_->GetGossipHeardBy(npc_id, 0.2f);
    std::sort(gossip.begin(), gossip.end(), [](const auto& a, const auto& b) {
        return a.credibility > b.credibility;
    });

    nlohmann::json gossip_items = nlohmann::json::array();
    for (size_t i = 0; i < std::min<size_t>(gossip.size(), 5); ++i) {
        const auto& g = gossip[i];
        nlohmann::json gi;
        gi["content"] = g.content;
        gi["about"] = g.about_entity;
        gi["source"] = g.source_npc;
        gi["credibility"] = g.credibility;
        gi["emotional_charge"] = g.emotional_charge;
        gossip_items.push_back(gi);
    }
    context["gossip"] = gossip_items;
    
    // Get allies and enemies
    auto allies = social_fabric_network_->GetAllies(npc_id, 0.5f);
    auto enemies = social_fabric_network_->GetEnemies(npc_id, 0.5f);
    context["allies"] = allies;
    context["enemies"] = enemies;
    
    // === 3. Emotional Continuity Context ===
    
    auto current_emotion = emotional_continuity_system_->GetCurrentEmotion();
    nlohmann::json emotion;
    emotion["joy"] = current_emotion.joy;
    emotion["trust"] = current_emotion.trust;
    emotion["fear"] = current_emotion.fear;
    emotion["sadness"] = current_emotion.sadness;
    emotion["anger"] = current_emotion.anger;
    emotion["surprise"] = current_emotion.surprise;
    emotion["valence"] = current_emotion.GetValence();
    emotion["arousal"] = current_emotion.GetArousal();
    emotion["dominant"] = emotional_continuity_system_->GetDominantEmotion();
    emotion["description"] = emotional_continuity_system_->DescribeEmotion();
    context["current_emotion"] = emotion;
    
    // Get personality
    auto personality = emotional_continuity_system_->GetPersonality();
    nlohmann::json pers;
    pers["openness"] = personality.openness;
    pers["conscientiousness"] = personality.conscientiousness;
    pers["extraversion"] = personality.extraversion;
    pers["agreeableness"] = personality.agreeableness;
    pers["neuroticism"] = personality.neuroticism;
    context["personality"] = pers;
    
    // Get sentiments toward entities
    auto sentiments = emotional_continuity_system_->GetAllSentiments();
    nlohmann::json sents = nlohmann::json::array();
    for (const auto& sent : sentiments) {
        nlohmann::json s;
        s["entity"] = sent.entity_id;
        s["sentiment"] = sent.sentiment;
        s["intensity"] = sent.intensity;
        sents.push_back(s);
    }
    context["sentiments"] = sents;
    
    // === 4. Player Behavior Modeling Context ===
    if (player_behavior_modeling_) {
        auto stats = player_behavior_modeling_->GetStats();
        nlohmann::json behavior;
        behavior["total_actions"] = stats.total_actions;
        behavior["patterns_detected"] = stats.patterns_detected;
        behavior["avg_success_rate"] = stats.avg_success_rate;
        behavior["dominant_playstyle"] = stats.dominant_playstyle;
        
        // Engine: Inject strategic counters
        behavior["strategic_suggestion"] = player_behavior_modeling_->SuggestCounterStrategy("current_dialogue");
        
        context["player_behavior"] = behavior;
    }
    
    // === 5. Ambient Awareness Context ===
    if (ambient_awareness_system_) {
        auto stats = ambient_awareness_system_->GetStats();
        nlohmann::json awareness;
        awareness["direct_observations_count"] = stats.direct_observations;
        awareness["inferred_events_count"] = stats.inferred_events;
        
        // Engine: Inject top 5 most certain events
        auto events = ambient_awareness_system_->GetAllKnownEvents(0.3f);
        std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
            return a.certainty > b.certainty;
        });

        nlohmann::json event_list = nlohmann::json::array();
        for (size_t i = 0; i < std::min<size_t>(events.size(), 5); ++i) {
            const auto& ev = events[i];
            nlohmann::json e;
            e["type"] = ev.event_type;
            e["description"] = ev.description;
            e["witnessed"] = ev.directly_witnessed;
            e["certainty"] = ev.certainty;
            e["location"] = ev.location;
            event_list.push_back(e);
        }
        awareness["current_events"] = event_list;
        
        context["ambient_awareness"] = awareness;
    }
    
    return context;
}

} // namespace NPCInference
