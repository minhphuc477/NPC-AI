#include "SocialFabricNetwork.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace NPCInference;

void PrintSeparator() {
    std::cout << "\n" << std::string(70, '=') << "\n\n";
}

void SimulateTime(int seconds) {
    std::cout << "[Simulating " << seconds << " seconds...]\n";
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

int main() {
    std::cout << "=== Social Fabric Network Demo ===\n";
    std::cout << "Emergent NPC Relationships & Gossip Propagation\n";
    PrintSeparator();
    
    SocialFabricNetwork social;
    
    // === Act 1: Initial Relationships ===
    std::cout << "ACT 1: BUILDING RELATIONSHIPS\n";
    std::cout << "------------------------------\n\n";
    
    // Player helps Elara
    std::cout << "Event: Player saves Elara from bandits\n";
    social.UpdateRelationship("Elara", "Player", 0.8f, 0.6f, 0.7f, "bandit_rescue");
    std::cout << "  Elara → Player: +trust, +affection, +respect\n\n";
    
    // Elara and Marcus are old friends
    std::cout << "Event: Elara and Marcus have been friends for years\n";
    social.UpdateRelationship("Elara", "Marcus", 0.7f, 0.8f, 0.6f, "old_friends");
    social.UpdateRelationship("Marcus", "Elara", 0.7f, 0.7f, 0.5f, "old_friends");
    std::cout << "  Mutual friendship established\n\n";
    
    // Marcus distrusts Player initially
    std::cout << "Event: Marcus is suspicious of strangers\n";
    social.UpdateRelationship("Marcus", "Player", -0.3f, -0.1f, 0.0f);
    std::cout << "  Marcus → Player: -trust, -affection, neutral respect\n\n";
    
    // Lyra and Elara are acquaintances
    std::cout << "Event: Lyra and Elara know each other from the market\n";
    social.UpdateRelationship("Lyra", "Elara", 0.4f, 0.3f, 0.5f);
    social.UpdateRelationship("Elara", "Lyra", 0.5f, 0.4f, 0.6f);
    std::cout << "  Casual acquaintance relationship\n\n";
    
    PrintSeparator();
    
    // === Act 2: Gossip Starts ===
    std::cout << "ACT 2: GOSSIP BEGINS\n";
    std::cout << "--------------------\n\n";
    
    std::cout << "Elara starts gossip: 'The Player is a hero!'\n";
    std::string gossip1 = social.StartGossip(
        "Elara",
        "The Player saved me from bandits! They're incredibly brave.",
        "Player",
        0.9f,   // High credibility (she was there)
        0.8f    // Very positive
    );
    std::cout << "  Credibility: 0.9 | Emotional charge: +0.8\n\n";
    
    std::cout << "Propagating gossip through social network...\n";
    social.PropagateGossip(gossip1, 3, 0.2f);
    
    auto heard_by_marcus = social.GetGossipHeardBy("Marcus");
    if (!heard_by_marcus.empty()) {
        std::cout << "  ✓ Marcus heard the gossip (through Elara)\n";
    }
    
    auto heard_by_lyra = social.GetGossipHeardBy("Lyra");
    if (!heard_by_lyra.empty()) {
        std::cout << "  ✓ Lyra heard the gossip (through Elara)\n";
    }
    
    PrintSeparator();
    
    // === Act 3: Reputation Changes ===
    std::cout << "ACT 3: REPUTATION IMPACT\n";
    std::cout << "------------------------\n\n";
    
    std::cout << "Marcus's opinion of Player BEFORE gossip:\n";
    auto marcus_rel_before = social.GetRelationship("Marcus", "Player");
    std::cout << "  Trust: " << marcus_rel_before.trust << "\n";
    std::cout << "  Affection: " << marcus_rel_before.affection << "\n";
    std::cout << "  Respect: " << marcus_rel_before.respect << "\n\n";
    
    float marcus_rep = social.GetReputation("Player", "Marcus");
    std::cout << "Player's reputation from Marcus's perspective: " << marcus_rep << "\n";
    std::cout << "  (Influenced by Elara's gossip!)\n\n";
    
    // Marcus warms up to Player after hearing gossip
    std::cout << "Event: Marcus reconsiders after hearing Elara's story\n";
    social.UpdateRelationship("Marcus", "Player", 0.4f, 0.3f, 0.5f, "gossip_influence");
    std::cout << "  Marcus → Player: +trust, +affection, +respect\n\n";
    
    PrintSeparator();
    
    // === Act 4: Negative Gossip ===
    std::cout << "ACT 4: NEGATIVE GOSSIP\n";
    std::cout << "----------------------\n\n";
    
    std::cout << "Theron (guard captain) starts negative gossip about Player\n";
    std::string gossip2 = social.StartGossip(
        "Theron",
        "I don't trust that Player. They're hiding something.",
        "Player",
        0.6f,   // Moderate credibility
        -0.7f   // Negative
    );
    std::cout << "  Credibility: 0.6 | Emotional charge: -0.7\n\n";
    
    // Theron is friends with the town guard
    social.UpdateRelationship("Theron", "Guard_A", 0.8f, 0.6f, 0.7f);
    social.UpdateRelationship("Theron", "Guard_B", 0.7f, 0.5f, 0.8f);
    
    std::cout << "Propagating negative gossip...\n";
    social.PropagateGossip(gossip2, 2, 0.25f);
    
    auto all_gossip_about_player = social.GetGossipAbout("Player");
    std::cout << "\nAll gossip about Player:\n";
    for (const auto& g : all_gossip_about_player) {
        std::cout << "  • \"" << g.content << "\"\n";
        std::cout << "    From: " << g.source_npc 
                  << " | Charge: " << g.emotional_charge 
                  << " | Heard by " << g.heard_by.size() << " NPCs\n";
    }
    
    PrintSeparator();
    
    // === Act 5: Social Network Analysis ===
    std::cout << "ACT 5: SOCIAL NETWORK ANALYSIS\n";
    std::cout << "-------------------------------\n\n";
    
    // Player's allies
    auto player_allies = social.GetAllies("Player", 0.4f);
    std::cout << "Player's allies:\n";
    for (const auto& ally : player_allies) {
        auto rel = social.GetRelationship(ally, "Player");
        std::cout << "  • " << ally << " (strength: " << rel.GetStrength() << ")\n";
    }
    std::cout << "\n";
    
    // Elara's allies
    auto elara_allies = social.GetAllies("Elara", 0.4f);
    std::cout << "Elara's allies:\n";
    for (const auto& ally : elara_allies) {
        auto rel = social.GetRelationship("Elara", ally);
        std::cout << "  • " << ally << " (strength: " << rel.GetStrength() << ")\n";
    }
    std::cout << "\n";
    
    // Mutual friends
    auto mutual = social.GetMutualFriends("Player", "Marcus");
    std::cout << "Mutual friends between Player and Marcus:\n";
    for (const auto& friend_id : mutual) {
        std::cout << "  • " << friend_id << "\n";
    }
    if (mutual.empty()) {
        std::cout << "  (None yet - but Elara connects them!)\n";
    }
    std::cout << "\n";
    
    // Social distance
    int distance = social.GetSocialDistance("Player", "Lyra");
    std::cout << "Social distance Player → Lyra: " << distance << " degrees\n";
    std::cout << "  (Player → Elara → Lyra)\n\n";
    
    PrintSeparator();
    
    // === Act 6: Faction Detection ===
    std::cout << "ACT 6: EMERGENT FACTIONS\n";
    std::cout << "------------------------\n\n";
    
    // Add more NPCs to create factions
    social.UpdateRelationship("Guard_A", "Guard_B", 0.9f, 0.7f, 0.8f);
    social.UpdateRelationship("Guard_B", "Guard_A", 0.9f, 0.7f, 0.8f);
    social.UpdateRelationship("Guard_A", "Theron", 0.8f, 0.6f, 0.9f);
    social.UpdateRelationship("Guard_B", "Theron", 0.8f, 0.6f, 0.9f);
    
    auto factions = social.DetectFactions();
    std::cout << "Detected " << factions.size() << " faction(s):\n\n";
    
    for (size_t i = 0; i < factions.size(); i++) {
        std::cout << "Faction " << (i+1) << ":\n";
        for (const auto& member : factions[i]) {
            std::cout << "  • " << member << "\n";
        }
        std::cout << "\n";
    }
    
    PrintSeparator();
    
    // === Act 7: Social Influence ===
    std::cout << "ACT 7: OPINION SPREADING\n";
    std::cout << "------------------------\n\n";
    
    std::cout << "Elara forms opinion: 'The old ruins are dangerous' (strength: 0.8)\n";
    social.SpreadOpinion("Elara", "old_ruins_danger", 0.8f);
    std::cout << "  Opinion spreads to friends...\n\n";
    
    float marcus_opinion = social.GetInfluencedOpinion("Marcus", "old_ruins_danger");
    std::cout << "Marcus's influenced opinion on 'old_ruins_danger': " << marcus_opinion << "\n";
    std::cout << "  (Influenced by friend Elara)\n\n";
    
    float lyra_opinion = social.GetInfluencedOpinion("Lyra", "old_ruins_danger");
    std::cout << "Lyra's influenced opinion on 'old_ruins_danger': " << lyra_opinion << "\n";
    std::cout << "  (Weaker influence - less close to Elara)\n\n";
    
    PrintSeparator();
    
    // === Statistics ===
    std::cout << "SOCIAL NETWORK STATISTICS\n";
    std::cout << "-------------------------\n\n";
    
    auto stats = social.GetStats();
    std::cout << "Total Relationships: " << stats.total_relationships << "\n";
    std::cout << "Positive Relationships: " << stats.positive_relationships << "\n";
    std::cout << "Negative Relationships: " << stats.negative_relationships << "\n";
    std::cout << "Average Relationship Strength: " << stats.avg_relationship_strength << "\n";
    std::cout << "Total Gossip Items: " << stats.total_gossip_items << "\n";
    std::cout << "Active Gossip Items: " << stats.active_gossip_items << "\n";
    std::cout << "Detected Factions: " << stats.detected_factions << "\n\n";
    
    PrintSeparator();
    
    // === Persistence ===
    std::cout << "PERSISTENCE TEST\n";
    std::cout << "----------------\n\n";
    
    std::string save_path = "social_network.json";
    if (social.Save(save_path)) {
        std::cout << "✓ Saved social network to " << save_path << "\n";
    }
    
    SocialFabricNetwork loaded;
    if (loaded.Load(save_path)) {
        std::cout << "✓ Loaded social network from " << save_path << "\n\n";
        
        auto loaded_stats = loaded.GetStats();
        std::cout << "Loaded " << loaded_stats.total_relationships << " relationships\n";
        std::cout << "Loaded " << loaded_stats.total_gossip_items << " gossip items\n";
    }
    
    PrintSeparator();
    
    std::cout << "✅ DEMO COMPLETE\n\n";
    std::cout << "Key Takeaways:\n";
    std::cout << "1. Relationships form dynamically (trust, affection, respect)\n";
    std::cout << "2. Gossip propagates through social networks\n";
    std::cout << "3. Reputation influenced by direct relationship + gossip + friends\n";
    std::cout << "4. Factions emerge from strong relationship clusters\n";
    std::cout << "5. Opinions spread through social influence\n";
    std::cout << "6. Full persistence for long-term social dynamics\n\n";
    std::cout << "This solves the unsolved problem of isolated NPCs!\n";
    std::cout << "NPCs now have EMERGENT social structures.\n";
    
    return 0;
}
