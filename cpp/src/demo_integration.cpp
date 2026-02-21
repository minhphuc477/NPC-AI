#include "NPCInference.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace NPCInference;

int main() {
    std::cout << "=== NPC AI Integration Demo ===\n";
    std::cout << "Demonstrating all 3 innovations working together\n\n";
    
    // Create NPC engine
    NPCInferenceEngine engine;
    
    // Start a conversation to initialize cognitive systems
    std::string session_id = engine.StartConversation("Elara", "Player");
    std::cout << "✓ Initialized NPC 'Elara' with cognitive systems\n\n";
    
    // === Setup NPC Personality ===
    std::cout << "SETUP: Configuring Elara's personality\n";
    std::cout << "---------------------------------------\n";
    
    auto* emotional = engine.GetEmotionalContinuity();
    if (emotional) {
        emotional->SetPersonality(Personalities::Cheerful());
        std::cout << "✓ Set cheerful personality (high agreeableness, low neuroticism)\n";
    }
    
    // === Scenario 1: First Meeting ===
    std::cout << "\n=== SCENARIO 1: First Meeting ===\n\n";
    
    auto* temporal = engine.GetTemporalMemory();
    if (temporal) {
        temporal->AddEpisode(
            "Met the Player for the first time at the market",
            0.1f,  // valence (slightly positive)
            0.3f,  // arousal
            0.6f   // importance
        );
        std::cout << "✓ Added episodic memory: First meeting\n";
    }
    
    auto* social = engine.GetSocialFabric();
    if (social) {
        social->UpdateRelationship("Elara", "Player", 0.3f, 0.2f, 0.1f, "first_meeting");
        std::cout << "✓ Updated relationship: Slight positive impression\n";
    }
    
    if (emotional) {
        EmotionalState curious;
        curious.surprise = 0.4f;
        curious.anticipation = 0.5f;
        emotional->ApplyEmotionalStimulus(curious, 0.6f);
        std::cout << "✓ Applied emotion: Curious and anticipating\n";
    }
    
    // Build context
    auto context1 = engine.BuildAdvancedContext("Elara", "Who is the Player?");
    std::cout << "\n📊 Advanced Context Generated:\n";
    std::cout << "  Memories: " << context1["memories"].size() << "\n";
    std::cout << "  Relationships: " << context1["relationships"].size() << "\n";
    std::cout << "  Current emotion: " << context1["current_emotion"]["description"].get<std::string>() << "\n";
    
    // === Scenario 2: Player Helps Elara ===
    std::cout << "\n\n=== SCENARIO 2: Player Saves Elara from Bandits ===\n\n";
    
    if (temporal) {
        temporal->AddEpisode(
            "The Player bravely saved me from bandits!",
            0.9f,   // Very positive valence
            0.9f,   // Very high arousal
            0.95f   // Very important!
        );
        std::cout << "✓ Added episodic memory: Bandit rescue (high importance + emotion)\n";
    }
    
    if (social) {
        social->UpdateRelationship("Elara", "Player", 0.7f, 0.8f, 0.6f, "bandit_rescue");
        std::cout << "✓ Updated relationship: Major increase in trust, affection, respect\n";
    }
    
    if (emotional) {
        emotional->UpdateSentiment("Player", 0.8f, 0.9f, "bandit_rescue");
        
        EmotionalState grateful;
        grateful.joy = 0.8f;
        grateful.trust = 0.7f;
        emotional->ApplyEmotionalStimulus(grateful, 1.0f);
        std::cout << "✓ Updated sentiment: Strong positive feelings toward Player\n";
        std::cout << "✓ Applied emotion: Grateful and joyful\n";
    }
    
    // Elara starts gossip about the heroic Player
    if (social) {
        std::string gossip_id = social->StartGossip(
            "Elara",
            "The Player is a true hero! They saved me from bandits!",
            "Player",
            0.95f,  // Very credible (she was there)
            0.9f    // Very positive
        );
        social->PropagateGossip(gossip_id, 2, 0.2f);
        std::cout << "✓ Started gossip: Spreading word of Player's heroism\n";
    }
    
    auto context2 = engine.BuildAdvancedContext("Elara", "How do you feel about the Player?");
    std::cout << "\n📊 Advanced Context After Rescue:\n";
    std::cout << "  Memories: " << context2["memories"].size() << "\n";
    std::cout << "  Relationships: " << context2["relationships"].size() << "\n";
    if (!context2["relationships"].empty()) {
        auto rel = context2["relationships"][0];
        std::cout << "    → Player: trust=" << rel["trust"].get<float>() 
                  << ", affection=" << rel["affection"].get<float>() << "\n";
    }
    std::cout << "  Sentiments: " << context2["sentiments"].size() << "\n";
    if (!context2["sentiments"].empty()) {
        auto sent = context2["sentiments"][0];
        std::cout << "    → Player: sentiment=" << sent["sentiment"].get<float>() 
                  << ", intensity=" << sent["intensity"].get<float>() << "\n";
    }
    std::cout << "  Current emotion: " << context2["current_emotion"]["description"].get<std::string>() << "\n";
    std::cout << "  Gossip items: " << context2["gossip"].size() << "\n";
    
    // === Scenario 3: Time Passes ===
    std::cout << "\n\n=== SCENARIO 3: Time Passes (Memory Decay) ===\n\n";
    
    if (temporal) {
        // Simulate time passing
        for (int i = 0; i < 5; i++) {
            temporal->UpdateMemoryStrengths();  // Update decay
        }
        std::cout << "✓ Simulated 5 days passing\n";
        
        auto memories = temporal->RetrieveEpisodes("Player", 10, 0.1f);
        std::cout << "  Memories retrieved: " << memories.size() << "\n";
        for (const auto& mem : memories) {
            std::cout << "    • \"" << mem.description << "\" (strength: " 
                      << mem.current_strength << ")\n";
        }
        std::cout << "  💡 Emotional memory (bandit rescue) decays slower!\n";
    }
    
    if (emotional) {
        // Emotions decay toward baseline
        for (int i = 0; i < 3; i++) {
            emotional->DecayTowardBaseline(1.0f);
        }
        std::cout << "✓ Emotions decayed toward baseline\n";
        std::cout << "  Current: " << emotional->DescribeEmotion() << "\n";
    }
    
    // === Final Context ===
    std::cout << "\n\n=== FINAL INTEGRATED CONTEXT ===\n\n";
    
    auto final_context = engine.BuildAdvancedContext("Elara", "Tell me about your experiences");
    
    std::cout << "Complete NPC State:\n";
    std::cout << "-------------------\n";
    std::cout << "Personality:\n";
    auto pers = final_context["personality"];
    std::cout << "  Agreeableness: " << pers["agreeableness"].get<float>() << "\n";
    std::cout << "  Neuroticism: " << pers["neuroticism"].get<float>() << "\n\n";
    
    std::cout << "Current Emotion:\n";
    auto emo = final_context["current_emotion"];
    std::cout << "  " << emo["description"].get<std::string>() << "\n";
    std::cout << "  Valence: " << emo["valence"].get<float>() << "\n\n";
    
    std::cout << "Memories (" << final_context["memories"].size() << "):\n";
    for (const auto& mem : final_context["memories"]) {
        std::cout << "  • " << mem["content"].get<std::string>() 
                  << " (strength: " << mem["strength"].get<float>() << ")\n";
    }
    std::cout << "\n";
    
    std::cout << "Relationships (" << final_context["relationships"].size() << "):\n";
    for (const auto& rel : final_context["relationships"]) {
        std::cout << "  • " << rel["entity"].get<std::string>() 
                  << ": trust=" << rel["trust"].get<float>()
                  << ", affection=" << rel["affection"].get<float>() << "\n";
    }
    std::cout << "\n";
    
    std::cout << "✅ INTEGRATION DEMO COMPLETE\n\n";
    std::cout << "All 3 innovation systems working together:\n";
    std::cout << "1. ✅ Temporal Memory: Memories decay realistically\n";
    std::cout << "2. ✅ Social Fabric: Relationships and gossip tracked\n";
    std::cout << "3. ✅ Emotional Continuity: Persistent emotions with personality\n\n";
    std::cout << "NPCs now have MEMORY, RELATIONSHIPS, and EMOTIONS!\n";
    std::cout << "This architecture is ready to use in your game.\n";
    
    return 0;
}
