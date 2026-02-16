#include "SocialFabricNetwork.h"
#include <algorithm>
#include <random>
#include <queue>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>

namespace NPCInference {

SocialFabricNetwork::SocialFabricNetwork() {
    // Initialize empty network
}

void SocialFabricNetwork::UpdateRelationship(
    const std::string& npc_a,
    const std::string& npc_b,
    float trust_delta,
    float affection_delta,
    float respect_delta,
    const std::string& experience_id
) {
    std::string key = MakeRelationshipKey(npc_a, npc_b);
    
    // Get or create relationship
    Relationship& rel = relationships_[key];
    rel.npc_a = npc_a;
    rel.npc_b = npc_b;
    
    // Update dimensions (clamp to -1, 1)
    rel.trust = std::clamp(rel.trust + trust_delta, -1.0f, 1.0f);
    rel.affection = std::clamp(rel.affection + affection_delta, -1.0f, 1.0f);
    rel.respect = std::clamp(rel.respect + respect_delta, -1.0f, 1.0f);
    
    // Update metadata
    if (rel.first_met == 0) {
        rel.first_met = GetCurrentTimestamp();
    }
    rel.last_interaction = GetCurrentTimestamp();
    rel.interaction_count++;
    
    if (!experience_id.empty()) {
        rel.shared_experiences.push_back(experience_id);
    }
}

Relationship SocialFabricNetwork::GetRelationship(
    const std::string& npc_a,
    const std::string& npc_b
) const {
    std::string key = MakeRelationshipKey(npc_a, npc_b);
    auto it = relationships_.find(key);
    
    if (it != relationships_.end()) {
        return it->second;
    }
    
    // Return neutral relationship if not found
    Relationship neutral;
    neutral.npc_a = npc_a;
    neutral.npc_b = npc_b;
    return neutral;
}

std::vector<Relationship> SocialFabricNetwork::GetAllRelationships(
    const std::string& npc_id
) const {
    std::vector<Relationship> results;
    
    for (const auto& [key, rel] : relationships_) {
        if (rel.npc_a == npc_id) {
            results.push_back(rel);
        }
    }
    
    return results;
}

std::string SocialFabricNetwork::StartGossip(
    const std::string& source_npc,
    const std::string& content,
    const std::string& about_entity,
    float credibility,
    float emotional_charge
) {
    GossipItem gossip;
    gossip.gossip_id = GenerateGossipId();
    gossip.content = content;
    gossip.source_npc = source_npc;
    gossip.about_entity = about_entity;
    gossip.credibility = std::clamp(credibility, 0.0f, 1.0f);
    gossip.emotional_charge = std::clamp(emotional_charge, -1.0f, 1.0f);
    gossip.timestamp = GetCurrentTimestamp();
    gossip.heard_by.insert(source_npc);
    gossip.current_strength = 1.0f;
    
    gossip_items_[gossip.gossip_id] = gossip;
    
    return gossip.gossip_id;
}

void SocialFabricNetwork::PropagateGossip(
    const std::string& gossip_id,
    int max_hops,
    float decay_rate
) {
    auto it = gossip_items_.find(gossip_id);
    if (it == gossip_items_.end()) return;
    
    GossipItem& gossip = it->second;
    std::set<std::string> visited;
    
    // Start BFS from source
    PropagateGossipBFS(gossip_id, gossip.source_npc, max_hops, 1.0f, visited);
}

void SocialFabricNetwork::PropagateGossipBFS(
    const std::string& gossip_id,
    const std::string& current_npc,
    int hops_remaining,
    float current_strength,
    std::set<std::string>& visited
) {
    if (hops_remaining <= 0 || current_strength < 0.1f) return;
    if (visited.count(current_npc)) return;
    
    visited.insert(current_npc);
    
    auto& gossip = gossip_items_[gossip_id];
    gossip.heard_by.insert(current_npc);
    gossip.propagation_count++;
    
    // Find friends to spread to
    auto relationships = GetAllRelationships(current_npc);
    for (const auto& rel : relationships) {
        if (!rel.IsPositive()) continue;  // Only spread to friends
        if (visited.count(rel.npc_b)) continue;
        
        // Strength decays based on relationship strength
        float new_strength = current_strength * (1.0f - gossip_decay_rate_) * rel.GetStrength();
        
        // Recursively propagate
        PropagateGossipBFS(gossip_id, rel.npc_b, hops_remaining - 1, new_strength, visited);
    }
}

std::vector<GossipItem> SocialFabricNetwork::GetGossipHeardBy(
    const std::string& npc_id,
    float min_strength
) const {
    std::vector<GossipItem> results;
    
    for (const auto& [id, gossip] : gossip_items_) {
        if (gossip.heard_by.count(npc_id) && gossip.current_strength >= min_strength) {
            results.push_back(gossip);
        }
    }
    
    // Sort by strength descending
    std::sort(results.begin(), results.end(),
        [](const GossipItem& a, const GossipItem& b) {
            return a.current_strength > b.current_strength;
        });
    
    return results;
}

std::vector<GossipItem> SocialFabricNetwork::GetGossipAbout(
    const std::string& entity,
    float min_strength
) const {
    std::vector<GossipItem> results;
    
    for (const auto& [id, gossip] : gossip_items_) {
        if (gossip.about_entity == entity && gossip.current_strength >= min_strength) {
            results.push_back(gossip);
        }
    }
    
    std::sort(results.begin(), results.end(),
        [](const GossipItem& a, const GossipItem& b) {
            return a.current_strength > b.current_strength;
        });
    
    return results;
}

float SocialFabricNetwork::GetReputation(
    const std::string& target_npc,
    const std::string& observer_npc
) const {
    float reputation = 0.0f;
    float weight_sum = 0.0f;
    
    // 1. Direct relationship (weight: 0.5)
    auto direct_rel = GetRelationship(observer_npc, target_npc);
    float direct_score = (direct_rel.trust + direct_rel.affection + direct_rel.respect) / 3.0f;
    reputation += direct_score * 0.5f;
    weight_sum += 0.5f;
    
    // 2. Gossip heard (weight: 0.3)
    auto gossip = GetGossipAbout(target_npc, 0.1f);
    float gossip_score = 0.0f;
    for (const auto& g : gossip) {
        if (g.heard_by.count(observer_npc)) {
            gossip_score += g.emotional_charge * g.credibility * g.current_strength;
        }
    }
    if (!gossip.empty()) {
        gossip_score /= gossip.size();
        reputation += gossip_score * 0.3f;
        weight_sum += 0.3f;
    }
    
    // 3. Friend opinions (weight: 0.2)
    auto friends = GetAllies(observer_npc, 0.5f);
    float friend_opinion = 0.0f;
    for (const auto& friend_id : friends) {
        auto friend_rel = GetRelationship(friend_id, target_npc);
        friend_opinion += (friend_rel.trust + friend_rel.affection + friend_rel.respect) / 3.0f;
    }
    if (!friends.empty()) {
        friend_opinion /= friends.size();
        reputation += friend_opinion * 0.2f;
        weight_sum += 0.2f;
    }
    
    if (weight_sum > 0.0f) {
        reputation /= weight_sum;
    }
    
    return std::clamp(reputation, -1.0f, 1.0f);
}

float SocialFabricNetwork::GetOverallReputation(const std::string& npc_id) const {
    std::set<std::string> all_npcs;
    
    // Collect all NPCs
    for (const auto& [key, rel] : relationships_) {
        all_npcs.insert(rel.npc_a);
        all_npcs.insert(rel.npc_b);
    }
    
    if (all_npcs.empty()) return 0.0f;
    
    float total_rep = 0.0f;
    int count = 0;
    
    for (const auto& observer : all_npcs) {
        if (observer == npc_id) continue;
        total_rep += GetReputation(npc_id, observer);
        count++;
    }
    
    return count > 0 ? total_rep / count : 0.0f;
}

std::vector<std::string> SocialFabricNetwork::GetAllies(
    const std::string& npc_id,
    float min_strength
) const {
    std::vector<std::string> allies;
    
    auto relationships = GetAllRelationships(npc_id);
    for (const auto& rel : relationships) {
        if (rel.IsPositive() && rel.GetStrength() >= min_strength) {
            allies.push_back(rel.npc_b);
        }
    }
    
    return allies;
}

std::vector<std::string> SocialFabricNetwork::GetEnemies(
    const std::string& npc_id,
    float min_strength
) const {
    std::vector<std::string> enemies;
    
    auto relationships = GetAllRelationships(npc_id);
    for (const auto& rel : relationships) {
        if (!rel.IsPositive() && rel.GetStrength() >= min_strength) {
            enemies.push_back(rel.npc_b);
        }
    }
    
    return enemies;
}

std::vector<std::string> SocialFabricNetwork::GetMutualFriends(
    const std::string& npc_a,
    const std::string& npc_b
) const {
    auto friends_a = GetAllies(npc_a, 0.3f);
    auto friends_b = GetAllies(npc_b, 0.3f);
    
    std::vector<std::string> mutual;
    std::set_intersection(
        friends_a.begin(), friends_a.end(),
        friends_b.begin(), friends_b.end(),
        std::back_inserter(mutual)
    );
    
    return mutual;
}

int SocialFabricNetwork::GetSocialDistance(
    const std::string& npc_a,
    const std::string& npc_b
) const {
    if (npc_a == npc_b) return 0;
    
    // BFS to find shortest path
    std::queue<std::pair<std::string, int>> q;
    std::set<std::string> visited;
    
    q.push({npc_a, 0});
    visited.insert(npc_a);
    
    while (!q.empty()) {
        auto [current, distance] = q.front();
        q.pop();
        
        if (current == npc_b) return distance;
        
        // Add all connected NPCs
        auto rels = GetAllRelationships(current);
        for (const auto& rel : rels) {
            if (!visited.count(rel.npc_b)) {
                visited.insert(rel.npc_b);
                q.push({rel.npc_b, distance + 1});
            }
        }
    }
    
    return -1;  // Not connected
}

std::vector<std::vector<std::string>> SocialFabricNetwork::DetectFactions() const {
    // Simple community detection: find strongly connected components
    std::vector<std::vector<std::string>> factions;
    std::set<std::string> all_npcs;
    std::set<std::string> assigned;
    
    // Collect all NPCs
    for (const auto& [key, rel] : relationships_) {
        all_npcs.insert(rel.npc_a);
        all_npcs.insert(rel.npc_b);
    }
    
    // For each unassigned NPC, find their faction
    for (const auto& npc : all_npcs) {
        if (assigned.count(npc)) continue;
        
        std::vector<std::string> faction;
        std::queue<std::string> q;
        q.push(npc);
        assigned.insert(npc);
        
        while (!q.empty()) {
            std::string current = q.front();
            q.pop();
            faction.push_back(current);
            
            // Add strong allies
            auto allies = GetAllies(current, 0.6f);
            for (const auto& ally : allies) {
                if (!assigned.count(ally)) {
                    assigned.insert(ally);
                    q.push(ally);
                }
            }
        }
        
        if (faction.size() >= 2) {  // Only count groups of 2+
            factions.push_back(faction);
        }
    }
    
    return factions;
}

void SocialFabricNetwork::SpreadOpinion(
    const std::string& npc_id,
    const std::string& opinion_content,
    float opinion_strength
) {
    std::string key = npc_id + ":" + opinion_content;
    opinions_[key] = std::clamp(opinion_strength, -1.0f, 1.0f);
    
    // Influence friends
    auto friends = GetAllies(npc_id, 0.4f);
    for (const auto& friend_id : friends) {
        std::string friend_key = friend_id + ":" + opinion_content;
        auto rel = GetRelationship(friend_id, npc_id);
        
        // Friend's opinion influenced by relationship strength
        float influence = opinion_strength * rel.GetStrength() * influence_strength_;
        
        if (opinions_.count(friend_key)) {
            // Blend with existing opinion
            opinions_[friend_key] = (opinions_[friend_key] + influence) / 2.0f;
        } else {
            opinions_[friend_key] = influence;
        }
    }
}

float SocialFabricNetwork::GetInfluencedOpinion(
    const std::string& npc_id,
    const std::string& topic
) const {
    std::string key = npc_id + ":" + topic;
    auto it = opinions_.find(key);
    
    if (it != opinions_.end()) {
        return it->second;
    }
    
    return 0.0f;  // Neutral if no opinion
}

bool SocialFabricNetwork::Save(const std::string& filepath) {
    nlohmann::json j;
    
    // Save relationships
    nlohmann::json rels = nlohmann::json::array();
    for (const auto& [key, rel] : relationships_) {
        nlohmann::json rel_json;
        rel_json["npc_a"] = rel.npc_a;
        rel_json["npc_b"] = rel.npc_b;
        rel_json["trust"] = rel.trust;
        rel_json["affection"] = rel.affection;
        rel_json["respect"] = rel.respect;
        rel_json["shared_experiences"] = rel.shared_experiences;
        rel_json["first_met"] = rel.first_met;
        rel_json["last_interaction"] = rel.last_interaction;
        rel_json["interaction_count"] = rel.interaction_count;
        rels.push_back(rel_json);
    }
    j["relationships"] = rels;
    
    // Save gossip
    nlohmann::json gossips = nlohmann::json::array();
    for (const auto& [id, gossip] : gossip_items_) {
        nlohmann::json g_json;
        g_json["id"] = gossip.gossip_id;
        g_json["content"] = gossip.content;
        g_json["source"] = gossip.source_npc;
        g_json["about"] = gossip.about_entity;
        g_json["credibility"] = gossip.credibility;
        g_json["emotional_charge"] = gossip.emotional_charge;
        g_json["timestamp"] = gossip.timestamp;
        g_json["heard_by"] = std::vector<std::string>(gossip.heard_by.begin(), gossip.heard_by.end());
        g_json["propagation_count"] = gossip.propagation_count;
        g_json["strength"] = gossip.current_strength;
        gossips.push_back(g_json);
    }
    j["gossip"] = gossips;
    
    // Save opinions
    nlohmann::json ops = nlohmann::json::object();
    for (const auto& [key, value] : opinions_) {
        ops[key] = value;
    }
    j["opinions"] = ops;
    
    // Save config
    j["config"]["gossip_decay_rate"] = gossip_decay_rate_;
    j["config"]["relationship_decay_rate"] = relationship_decay_rate_;
    j["config"]["influence_strength"] = influence_strength_;
    
    std::ofstream file(filepath);
    if (!file.is_open()) return false;
    
    file << std::setw(2) << j << std::endl;
    return true;
}

bool SocialFabricNetwork::Load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;
    
    nlohmann::json j;
    try {
        file >> j;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return false;
    }
    
    relationships_.clear();
    gossip_items_.clear();
    opinions_.clear();
    
    // Load relationships
    if (j.contains("relationships")) {
        for (const auto& rel_json : j["relationships"]) {
            Relationship rel;
            rel.npc_a = rel_json.value("npc_a", "");
            rel.npc_b = rel_json.value("npc_b", "");
            rel.trust = rel_json.value("trust", 0.0f);
            rel.affection = rel_json.value("affection", 0.0f);
            rel.respect = rel_json.value("respect", 0.0f);
            rel.shared_experiences = rel_json.value("shared_experiences", std::vector<std::string>{});
            rel.first_met = rel_json.value("first_met", 0L);
            rel.last_interaction = rel_json.value("last_interaction", 0L);
            rel.interaction_count = rel_json.value("interaction_count", 0);
            
            std::string key = MakeRelationshipKey(rel.npc_a, rel.npc_b);
            relationships_[key] = rel;
        }
    }
    
    // Load gossip
    if (j.contains("gossip")) {
        for (const auto& g_json : j["gossip"]) {
            GossipItem gossip;
            gossip.gossip_id = g_json.value("id", "");
            gossip.content = g_json.value("content", "");
            gossip.source_npc = g_json.value("source", "");
            gossip.about_entity = g_json.value("about", "");
            gossip.credibility = g_json.value("credibility", 0.5f);
            gossip.emotional_charge = g_json.value("emotional_charge", 0.0f);
            gossip.timestamp = g_json.value("timestamp", 0L);
            
            auto heard_vec = g_json.value("heard_by", std::vector<std::string>{});
            gossip.heard_by = std::set<std::string>(heard_vec.begin(), heard_vec.end());
            gossip.propagation_count = g_json.value("propagation_count", 0);
            gossip.current_strength = g_json.value("strength", 1.0f);
            
            gossip_items_[gossip.gossip_id] = gossip;
        }
    }
    
    // Load opinions
    if (j.contains("opinions")) {
        for (auto it = j["opinions"].begin(); it != j["opinions"].end(); ++it) {
            opinions_[it.key()] = it.value();
        }
    }
    
    // Load config
    if (j.contains("config")) {
        gossip_decay_rate_ = j["config"].value("gossip_decay_rate", 0.3f);
        relationship_decay_rate_ = j["config"].value("relationship_decay_rate", 0.01f);
        influence_strength_ = j["config"].value("influence_strength", 0.3f);
    }
    
    return true;
}

SocialFabricNetwork::SocialStats SocialFabricNetwork::GetStats() const {
    SocialStats stats;
    stats.total_relationships = relationships_.size();
    stats.positive_relationships = 0;
    stats.negative_relationships = 0;
    stats.avg_relationship_strength = 0.0f;
    
    for (const auto& [key, rel] : relationships_) {
        if (rel.IsPositive()) {
            stats.positive_relationships++;
        } else {
            stats.negative_relationships++;
        }
        stats.avg_relationship_strength += rel.GetStrength();
    }
    
    if (stats.total_relationships > 0) {
        stats.avg_relationship_strength /= stats.total_relationships;
    }
    
    stats.total_gossip_items = gossip_items_.size();
    stats.active_gossip_items = 0;
    for (const auto& [id, gossip] : gossip_items_) {
        if (gossip.current_strength > 0.1f) {
            stats.active_gossip_items++;
        }
    }
    
    stats.detected_factions = DetectFactions().size();
    
    return stats;
}

// === Private Helper Functions ===

std::string SocialFabricNetwork::MakeRelationshipKey(
    const std::string& a,
    const std::string& b
) const {
    return a + ":" + b;
}

std::string SocialFabricNetwork::GenerateGossipId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    const char* hex_chars = "0123456789abcdef";
    std::string id = "gossip_";
    for (int i = 0; i < 16; ++i) {
        id += hex_chars[dis(gen)];
    }
    return id;
}

int64_t SocialFabricNetwork::GetCurrentTimestamp() const {
    return std::chrono::system_clock::now().time_since_epoch().count() / 1000000000;
}

} // namespace NPCInference
