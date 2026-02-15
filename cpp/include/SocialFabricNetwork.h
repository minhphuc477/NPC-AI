#pragma once

#include <string>
#include <map>
#include <vector>
#include <set>
#include <memory>
#include <functional>
#include <nlohmann/json.hpp>

namespace NPCInference {

/**
 * Social Fabric Network - Emergent Social Dynamics for NPCs
 * 
 * Features:
 * - Dynamic Relationships: Trust, affection, respect that evolve over time
 * - Gossip Propagation: Information spreads through the NPC network
 * - Reputation System: Track how NPCs are perceived by others
 * - Faction Detection: Identify ally/enemy groups automatically
 * - Social Influence: Relationships affect NPC behavior
 */

struct Relationship {
    std::string npc_a;
    std::string npc_b;
    
    // Relationship dimensions (all -1 to 1)
    float trust = 0.0f;        // How much A trusts B
    float affection = 0.0f;    // How much A likes B
    float respect = 0.0f;      // How much A respects B
    
    // Relationship metadata
    std::vector<std::string> shared_experiences;  // Event IDs
    int64_t first_met = 0;
    int64_t last_interaction = 0;
    int interaction_count = 0;
    
    // Relationship strength (derived from dimensions)
    float GetStrength() const {
        return (std::abs(trust) + std::abs(affection) + std::abs(respect)) / 3.0f;
    }
    
    // Is this a positive relationship?
    bool IsPositive() const {
        return (trust + affection + respect) > 0.0f;
    }
};

struct GossipItem {
    std::string gossip_id;
    std::string content;           // What is being gossiped about
    std::string source_npc;        // Who started the gossip
    std::string about_entity;      // Who/what is it about
    float credibility = 0.5f;      // How believable (0 to 1)
    float emotional_charge = 0.0f; // How emotionally charged (-1 to 1)
    int64_t timestamp;
    
    // Propagation tracking
    std::set<std::string> heard_by;  // Which NPCs have heard this
    int propagation_count = 0;
    float current_strength = 1.0f;   // Decays as it spreads
};

class SocialFabricNetwork {
public:
    SocialFabricNetwork();
    ~SocialFabricNetwork() = default;
    
    // === Relationship Management ===
    
    /**
     * Update relationship between two NPCs based on an interaction
     * @param npc_a First NPC
     * @param npc_b Second NPC
     * @param trust_delta Change in trust (-1 to 1)
     * @param affection_delta Change in affection (-1 to 1)
     * @param respect_delta Change in respect (-1 to 1)
     * @param experience_id Optional event that caused this change
     */
    void UpdateRelationship(
        const std::string& npc_a,
        const std::string& npc_b,
        float trust_delta,
        float affection_delta,
        float respect_delta,
        const std::string& experience_id = ""
    );
    
    /**
     * Get relationship from A's perspective toward B
     */
    Relationship GetRelationship(
        const std::string& npc_a,
        const std::string& npc_b
    ) const;
    
    /**
     * Get all relationships for an NPC
     */
    std::vector<Relationship> GetAllRelationships(
        const std::string& npc_id
    ) const;
    
    // === Gossip Propagation ===
    
    /**
     * Start a new gossip
     * @param source_npc Who is starting the gossip
     * @param content What they're saying
     * @param about_entity Who/what it's about
     * @param credibility How believable (0 to 1)
     * @param emotional_charge How emotional (-1 to 1)
     * @return Gossip ID
     */
    std::string StartGossip(
        const std::string& source_npc,
        const std::string& content,
        const std::string& about_entity,
        float credibility = 0.5f,
        float emotional_charge = 0.0f
    );
    
    /**
     * Propagate gossip through the social network
     * @param gossip_id The gossip to propagate
     * @param max_hops Maximum propagation distance
     * @param decay_rate How much strength decays per hop
     */
    void PropagateGossip(
        const std::string& gossip_id,
        int max_hops = 3,
        float decay_rate = 0.3f
    );
    
    /**
     * Get all gossip an NPC has heard
     */
    std::vector<GossipItem> GetGossipHeardBy(
        const std::string& npc_id,
        float min_strength = 0.1f
    ) const;
    
    /**
     * Get all gossip about a specific entity
     */
    std::vector<GossipItem> GetGossipAbout(
        const std::string& entity,
        float min_strength = 0.1f
    ) const;
    
    // === Reputation System ===
    
    /**
     * Calculate reputation of target_npc from observer_npc's perspective
     * Considers: direct relationship + gossip heard + friend opinions
     */
    float GetReputation(
        const std::string& target_npc,
        const std::string& observer_npc
    ) const;
    
    /**
     * Get overall reputation (average across all NPCs)
     */
    float GetOverallReputation(const std::string& npc_id) const;
    
    // === Social Network Analysis ===
    
    /**
     * Get allies (NPCs with strong positive relationships)
     */
    std::vector<std::string> GetAllies(
        const std::string& npc_id,
        float min_strength = 0.5f
    ) const;
    
    /**
     * Get enemies (NPCs with strong negative relationships)
     */
    std::vector<std::string> GetEnemies(
        const std::string& npc_id,
        float min_strength = 0.5f
    ) const;
    
    /**
     * Find mutual friends between two NPCs
     */
    std::vector<std::string> GetMutualFriends(
        const std::string& npc_a,
        const std::string& npc_b
    ) const;
    
    /**
     * Calculate social distance (degrees of separation)
     */
    int GetSocialDistance(
        const std::string& npc_a,
        const std::string& npc_b
    ) const;
    
    /**
     * Detect emergent factions/cliques
     * Returns groups of NPCs with strong internal bonds
     */
    std::vector<std::vector<std::string>> DetectFactions() const;
    
    // === Social Influence ===
    
    /**
     * Spread opinion through social network
     * NPCs influence their friends' opinions
     * @param npc_id Who has the opinion
     * @param opinion_content What the opinion is about
     * @param opinion_strength How strong (-1 to 1)
     */
    void SpreadOpinion(
        const std::string& npc_id,
        const std::string& opinion_content,
        float opinion_strength
    );
    
    /**
     * Get influenced opinion (considering friend opinions)
     */
    float GetInfluencedOpinion(
        const std::string& npc_id,
        const std::string& topic
    ) const;
    
    // === Persistence ===
    
    bool Save(const std::string& filepath);
    bool Load(const std::string& filepath);
    
    // === Statistics ===
    
    struct SocialStats {
        int total_relationships;
        int positive_relationships;
        int negative_relationships;
        int total_gossip_items;
        int active_gossip_items;  // strength > 0.1
        float avg_relationship_strength;
        int detected_factions;
    };
    
    SocialStats GetStats() const;
    
    // === Configuration ===
    
    void SetGossipDecayRate(float rate) { gossip_decay_rate_ = rate; }
    void SetRelationshipDecayRate(float rate) { relationship_decay_rate_ = rate; }
    void SetInfluenceStrength(float strength) { influence_strength_ = strength; }
    
private:
    // Relationship storage: "npc_a:npc_b" -> Relationship
    std::map<std::string, Relationship> relationships_;
    
    // Gossip storage
    std::map<std::string, GossipItem> gossip_items_;
    
    // Opinion storage: "npc_id:topic" -> strength
    std::map<std::string, float> opinions_;
    
    // Configuration
    float gossip_decay_rate_ = 0.3f;
    float relationship_decay_rate_ = 0.01f;  // Very slow decay
    float influence_strength_ = 0.3f;
    
    // Helper functions
    std::string MakeRelationshipKey(const std::string& a, const std::string& b) const;
    std::string GenerateGossipId() const;
    int64_t GetCurrentTimestamp() const;
    
    // Graph traversal for propagation
    void PropagateGossipBFS(
        const std::string& gossip_id,
        const std::string& current_npc,
        int hops_remaining,
        float current_strength,
        std::set<std::string>& visited
    );
    
    // Faction detection (community detection algorithm)
    std::vector<std::vector<std::string>> DetectCommunitiesLouvain() const;
};

} // namespace NPCInference
