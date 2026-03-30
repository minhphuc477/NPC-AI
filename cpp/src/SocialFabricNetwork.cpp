#include "SocialFabricNetwork.h"
#include <algorithm>
#include <random>
#include <queue>
#include <cmath>
#include <fstream>
#include "NPCLogger.h"

namespace NPCInference {

SocialFabricNetwork::SocialFabricNetwork() {}

void SocialFabricNetwork::UpdateRelationship(const std::string& npc_a, const std::string& npc_b, float trust_delta, float affection_delta, float respect_delta, const std::string& experience_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = npc_a + ":" + npc_b;
    Relationship& relation_ref = relationships_[key];
    relation_ref.npc_a = npc_a;
    relation_ref.npc_b = npc_b;
    
    float t = relation_ref.trust + trust_delta;
    if (t < -1.0f) t = -1.0f;
    if (t > 1.0f) t = 1.0f;
    relation_ref.trust = t;
    
    float a = relation_ref.affection + affection_delta;
    if (a < -1.0f) a = -1.0f;
    if (a > 1.0f) a = 1.0f;
    relation_ref.affection = a;
    
    float r = relation_ref.respect + respect_delta;
    if (r < -1.0f) r = -1.0f;
    if (r > 1.0f) r = 1.0f;
    relation_ref.respect = r;
    
    if (!experience_id.empty()) {
        relation_ref.shared_experiences.push_back(experience_id);
    }
}

Relationship SocialFabricNetwork::GetRelationship(const std::string& npc_a, const std::string& npc_b) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = relationships_.find(npc_a + ":" + npc_b);
    if (it != relationships_.end()) return it->second;
    Relationship neutral;
    neutral.npc_a = npc_a;
    neutral.npc_b = npc_b;
    return neutral;
}

std::vector<Relationship> SocialFabricNetwork::GetAllRelationships(const std::string& npc_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Relationship> results;
    for (const auto& pair : relationships_) {
        if (pair.second.npc_a == npc_id) results.push_back(pair.second);
    }
    return results;
}

std::string SocialFabricNetwork::StartGossip(const std::string& source_npc, const std::string& content, const std::string& about_entity, float credibility, float emotional_charge) {
    std::lock_guard<std::mutex> lock(mutex_);
    GossipItem gossip;
    gossip.gossip_id = "gossip_" + std::to_string(std::rand());
    gossip.content = content;
    gossip.source_npc = source_npc;
    gossip.about_entity = about_entity;
    gossip.credibility = credibility;
    gossip.emotional_charge = emotional_charge;
    gossip.heard_by.insert(source_npc);
    gossip.current_strength = 1.0f;
    gossip_items_[gossip.gossip_id] = gossip;
    return gossip.gossip_id;
}

void SocialFabricNetwork::PropagateGossip(const std::string& gossip_id, int max_hops, float decay_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = gossip_items_.find(gossip_id);
    if (it == gossip_items_.end()) return;
    
    GossipItem& gossip = it->second;
    std::set<std::string> new_hearers;
    
    // Simulate propagation (simplified BFS)
    for (int hop = 0; hop < max_hops; ++hop) {
        if (gossip.current_strength < 0.1f) break;
        
        std::set<std::string> current_hop_hearers;
        for (const std::string& heard_by : gossip.heard_by) {
            // Find positive relationships (friends)
            for (const auto& [key, rel] : relationships_) {
                if (rel.npc_a == heard_by && rel.trust > 0.0f) {
                    if (gossip.heard_by.find(rel.npc_b) == gossip.heard_by.end()) {
                        current_hop_hearers.insert(rel.npc_b);
                    }
                }
            }
        }
        
        for (const std::string& new_hearer : current_hop_hearers) {
            gossip.heard_by.insert(new_hearer);
        }
        
        gossip.current_strength *= (1.0f - decay_rate);
    }
}

std::vector<GossipItem> SocialFabricNetwork::GetGossipHeardBy(const std::string& npc_id, float min_strength) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<GossipItem> results;
    for (const auto& pair : gossip_items_) {
        if (pair.second.heard_by.count(npc_id) && pair.second.current_strength >= min_strength) results.push_back(pair.second);
    }
    return results;
}

std::vector<GossipItem> SocialFabricNetwork::GetGossipAbout(const std::string& entity, float min_strength) const {
    std::vector<GossipItem> results;
    for (const auto& pair : gossip_items_) {
        if (pair.second.about_entity == entity && pair.second.current_strength >= min_strength) results.push_back(pair.second);
    }
    return results;
}

float SocialFabricNetwork::GetReputation(const std::string& target_npc, const std::string& observer_npc) const {
    Relationship rel = GetRelationship(observer_npc, target_npc);
    // Reputation from an observer's perspective is a blend of trust and respect
    return (rel.trust * 0.6f) + (rel.respect * 0.4f);
}

float SocialFabricNetwork::GetOverallReputation(const std::string& npc_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    float total_rep = 0.0f;
    int count = 0;
    
    for (const auto& [key, rel] : relationships_) {
        // Find relationships where the target is the receiver (npc_b)
        if (rel.npc_b == npc_id) {
            total_rep += (rel.trust * 0.6f) + (rel.respect * 0.4f);
            count++;
        }
    }
    
    return count > 0 ? (total_rep / count) : 0.0f;
}

std::vector<std::string> SocialFabricNetwork::GetAllies(const std::string& npc_id, float min_strength) const {
    std::vector<std::string> allies;
    for (const auto& rel : GetAllRelationships(npc_id)) {
        if (rel.IsPositive() && rel.GetStrength() >= min_strength) allies.push_back(rel.npc_b);
    }
    return allies;
}

std::vector<std::string> SocialFabricNetwork::GetEnemies(const std::string& npc_id, float min_strength) const {
    std::vector<std::string> enemies;
    for (const auto& rel : GetAllRelationships(npc_id)) {
        if (!rel.IsPositive() && rel.GetStrength() >= min_strength) enemies.push_back(rel.npc_b);
    }
    return enemies;
}

std::vector<std::string> SocialFabricNetwork::GetMutualFriends(const std::string& npc_a, const std::string& npc_b) const {
    std::vector<std::string> mutuals;
    std::vector<std::string> allies_a = GetAllies(npc_a, 0.3f); // Assuming 0.3 is the friend threshold
    std::vector<std::string> allies_b = GetAllies(npc_b, 0.3f);
    
    std::sort(allies_a.begin(), allies_a.end());
    std::sort(allies_b.begin(), allies_b.end());
    
    std::set_intersection(allies_a.begin(), allies_a.end(),
                          allies_b.begin(), allies_b.end(),
                          std::back_inserter(mutuals));
    
    return mutuals;
}

int SocialFabricNetwork::GetSocialDistance(const std::string& npc_a, const std::string& npc_b) const {
    if (npc_a == npc_b) return 0;
    
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<std::pair<std::string, int>> q;
    std::set<std::string> visited;
    
    q.push({npc_a, 0});
    visited.insert(npc_a);
    
    while (!q.empty()) {
        auto [current, dist] = q.front();
        q.pop();
        
        if (current == npc_b) return dist;
        if (dist >= 5) continue; // Max depth to search
        
        for (const auto& [key, rel] : relationships_) {
            if (rel.npc_a == current && rel.trust > 0.0f) {
                if (visited.find(rel.npc_b) == visited.end()) {
                    visited.insert(rel.npc_b);
                    q.push({rel.npc_b, dist + 1});
                }
            }
        }
    }
    
    return -1; // No path found
}

std::vector<std::vector<std::string>> SocialFabricNetwork::DetectFactions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::set<std::string> all_nodes;
    for (const auto& [key, rel] : relationships_) {
        all_nodes.insert(rel.npc_a);
        all_nodes.insert(rel.npc_b);
    }
    
    std::vector<std::vector<std::string>> factions;
    std::set<std::string> unvisited = all_nodes;
    
    while (!unvisited.empty()) {
        std::string start_node = *unvisited.begin();
        std::vector<std::string> current_faction;
        std::queue<std::string> q;
        
        q.push(start_node);
        unvisited.erase(start_node);
        
        while (!q.empty()) {
            std::string curr = q.front();
            q.pop();
            current_faction.push_back(curr);
            
            // Find all strongly connected friends
            for (const auto& [key, rel] : relationships_) {
                if (rel.npc_a == curr && rel.trust > 0.5f && rel.affection > 0.5f) { // Strong bond required
                    if (unvisited.find(rel.npc_b) != unvisited.end()) {
                        unvisited.erase(rel.npc_b);
                        q.push(rel.npc_b);
                    }
                }
                // Also check incoming edges for undirected factions
                if (rel.npc_b == curr && rel.trust > 0.5f && rel.affection > 0.5f) {
                    if (unvisited.find(rel.npc_a) != unvisited.end()) {
                        unvisited.erase(rel.npc_a);
                        q.push(rel.npc_a);
                    }
                }
            }
        }
        
        if (current_faction.size() > 1) { // A faction needs at least 2 people
            factions.push_back(current_faction);
        }
    }
    
    return factions;
}

void SocialFabricNetwork::SpreadOpinion(const std::string& npc_id, const std::string& opinion_content, float opinion_strength) {
    std::lock_guard<std::mutex> lock(mutex_);
    opinions_[npc_id + ":" + opinion_content] = opinion_strength;
}

float SocialFabricNetwork::GetInfluencedOpinion(const std::string& npc_id, const std::string& topic) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = opinions_.find(npc_id + ":" + topic);
    if (it != opinions_.end()) return it->second;
    return 0.0f;
}

bool SocialFabricNetwork::Save(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        nlohmann::json j;
        
        j["relationships"] = nlohmann::json::array();
        for (const auto& [key, rel] : relationships_) {
            nlohmann::json r;
            r["npc_a"] = rel.npc_a;
            r["npc_b"] = rel.npc_b;
            r["trust"] = rel.trust;
            r["affection"] = rel.affection;
            r["respect"] = rel.respect;
            r["shared_experiences"] = rel.shared_experiences;
            j["relationships"].push_back(r);
        }
        
        j["gossip_items"] = nlohmann::json::array();
        for (const auto& [id, gossip] : gossip_items_) {
            nlohmann::json g;
            g["gossip_id"] = gossip.gossip_id;
            g["content"] = gossip.content;
            g["source_npc"] = gossip.source_npc;
            g["about_entity"] = gossip.about_entity;
            g["credibility"] = gossip.credibility;
            g["emotional_charge"] = gossip.emotional_charge;
            g["current_strength"] = gossip.current_strength;
            g["heard_by"] = gossip.heard_by;
            j["gossip_items"].push_back(g);
        }
        
        std::ofstream file(filepath);
        if (!file.is_open()) return false;
        file << std::setw(4) << j << std::endl;
        return true;
    } catch (const std::exception& e) {
        NPCLogger::Error(std::string("Error saving SocialFabricNetwork: ") + e.what());
        return false;
    }
}

bool SocialFabricNetwork::Load(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        
        nlohmann::json j;
        file >> j;
        
        relationships_.clear();
        if (j.contains("relationships") && j["relationships"].is_array()) {
            for (const auto& item : j["relationships"]) {
                Relationship rel;
                rel.npc_a = item.value("npc_a", "");
                rel.npc_b = item.value("npc_b", "");
                rel.trust = item.value("trust", 0.0f);
                rel.affection = item.value("affection", 0.0f);
                rel.respect = item.value("respect", 0.0f);
                if (item.contains("shared_experiences") && item["shared_experiences"].is_array()) {
                    rel.shared_experiences = item["shared_experiences"].get<std::vector<std::string>>();
                }
                relationships_[MakeRelationshipKey(rel.npc_a, rel.npc_b)] = rel;
            }
        }
        
        gossip_items_.clear();
        if (j.contains("gossip_items") && j["gossip_items"].is_array()) {
            for (const auto& item : j["gossip_items"]) {
                GossipItem gossip;
                gossip.gossip_id = item.value("gossip_id", "");
                gossip.content = item.value("content", "");
                gossip.source_npc = item.value("source_npc", "");
                gossip.about_entity = item.value("about_entity", "");
                gossip.credibility = item.value("credibility", 0.0f);
                gossip.emotional_charge = item.value("emotional_charge", 0.0f);
                gossip.current_strength = item.value("current_strength", 1.0f);
                if (item.contains("heard_by") && item["heard_by"].is_array()) {
                    gossip.heard_by = item["heard_by"].get<std::set<std::string>>();
                }
                gossip_items_[gossip.gossip_id] = gossip;
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        NPCLogger::Error(std::string("Error loading SocialFabricNetwork: ") + e.what());
        return false;
    }
}

SocialFabricNetwork::SocialStats SocialFabricNetwork::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    SocialStats stats;
    stats.total_relationships = relationships_.size();
    stats.total_gossip_items = gossip_items_.size();
    
    float total_strength = 0.0f;
    for (const auto& [key, rel] : relationships_) {
        if (rel.IsPositive()) stats.positive_relationships++;
        else if (rel.GetStrength() > 0) stats.negative_relationships++;
        total_strength += rel.GetStrength();
    }
    
    if (!relationships_.empty()) {
        stats.avg_relationship_strength = total_strength / relationships_.size();
    }
    
    for (const auto& [id, gossip] : gossip_items_) {
        if (gossip.current_strength >= 0.1f) stats.active_gossip_items++;
    }
    
    return stats;
}

std::string SocialFabricNetwork::MakeRelationshipKey(const std::string& a, const std::string& b) const {
    return a + ":" + b;
}

std::string SocialFabricNetwork::GenerateGossipId() const {
    return "gossip_" + std::to_string(std::rand());
}

int64_t SocialFabricNetwork::GetCurrentTimestamp() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

} // namespace NPCInference
