#pragma once

#include <string>
#include <map>
#include <vector>
#include <set>
#include <memory>
#include <functional>
#include <mutex>
#include <nlohmann/json.hpp>

namespace NPCInference {

struct Relationship {
    std::string npc_a;
    std::string npc_b;
    float trust = 0.0f;
    float affection = 0.0f;
    float respect = 0.0f;
    std::vector<std::string> shared_experiences;
    int64_t first_met = 0;
    int64_t last_interaction = 0;
    int interaction_count = 0;
    float GetStrength() const { return (std::abs(trust) + std::abs(affection) + std::abs(respect)) / 3.0f; }
    bool IsPositive() const { return (trust + affection + respect) > 0.0f; }
};

struct GossipItem {
    std::string gossip_id;
    std::string content;
    std::string source_npc;
    std::string about_entity;
    float credibility = 0.5f;
    float emotional_charge = 0.0f;
    int64_t timestamp = 0;
    std::set<std::string> heard_by;
    int propagation_count = 0;
    float current_strength = 1.0f;
};

class SocialFabricNetwork {
public:
    SocialFabricNetwork();
    ~SocialFabricNetwork() = default;

    void UpdateRelationship(const std::string& npc_a, const std::string& npc_b, float trust_delta, float affection_delta, float respect_delta, const std::string& experience_id = "");
    Relationship GetRelationship(const std::string& npc_a, const std::string& npc_b) const;
    std::vector<Relationship> GetAllRelationships(const std::string& npc_id) const;
    
    std::string StartGossip(const std::string& source_npc, const std::string& content, const std::string& about_entity, float credibility = 0.5f, float emotional_charge = 0.0f);
    void PropagateGossip(const std::string& gossip_id, int max_hops = 3, float decay_rate = 0.3f);
    std::vector<GossipItem> GetGossipHeardBy(const std::string& npc_id, float min_strength = 0.1f) const;
    std::vector<GossipItem> GetGossipAbout(const std::string& entity, float min_strength = 0.1f) const;

    float GetReputation(const std::string& target_npc, const std::string& observer_npc) const;
    float GetOverallReputation(const std::string& npc_id) const;

    std::vector<std::string> GetAllies(const std::string& npc_id, float min_strength = 0.5f) const;
    std::vector<std::string> GetEnemies(const std::string& npc_id, float min_strength = 0.5f) const;
    std::vector<std::string> GetMutualFriends(const std::string& npc_a, const std::string& npc_b) const;
    int GetSocialDistance(const std::string& npc_a, const std::string& npc_b) const;
    std::vector<std::vector<std::string>> DetectFactions() const;

    void SpreadOpinion(const std::string& npc_id, const std::string& opinion_content, float opinion_strength);
    float GetInfluencedOpinion(const std::string& npc_id, const std::string& topic) const;

    bool Save(const std::string& filepath);
    bool Load(const std::string& filepath);

    struct SocialStats {
        int total_relationships = 0;
        int positive_relationships = 0;
        int negative_relationships = 0;
        int total_gossip_items = 0;
        int active_gossip_items = 0;
        float avg_relationship_strength = 0.0f;
        int detected_factions = 0;
    };
    SocialStats GetStats() const;

private:
    std::map<std::string, Relationship> relationships_;
    std::map<std::string, GossipItem> gossip_items_;
    std::map<std::string, float> opinions_;
    mutable std::mutex mutex_;
    float gossip_decay_rate_ = 0.3f;
    float relationship_decay_rate_ = 0.01f;
    float influence_strength_ = 0.3f;

    std::string MakeRelationshipKey(const std::string& a, const std::string& b) const;
    std::string GenerateGossipId() const;
    int64_t GetCurrentTimestamp() const;
};

} // namespace NPCInference
