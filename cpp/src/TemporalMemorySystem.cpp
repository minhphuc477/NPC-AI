#include "TemporalMemorySystem.h"
#include "NPCLogger.h"
#include <algorithm>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <set>

namespace NPCInference {

TemporalMemorySystem::TemporalMemorySystem() {
    // Initialize with current time
    UpdateMemoryStrengths();
}

std::string TemporalMemorySystem::AddEpisode(
    const std::string& description,
    float emotional_valence,
    float emotional_arousal,
    float importance,
    const std::vector<std::string>& participants,
    const std::string& location
) {
    std::lock_guard<std::mutex> lock(mutex_);
    EpisodicMemory memory;
    memory.event_id = GenerateMemoryId();
    memory.description = description;
    memory.timestamp = GetCurrentTimestamp();
    memory.emotional_valence = (emotional_valence < -1.0f) ? -1.0f : ((emotional_valence > 1.0f) ? 1.0f : emotional_valence);
    memory.emotional_arousal = (emotional_arousal < 0.0f) ? 0.0f : ((emotional_arousal > 1.0f) ? 1.0f : emotional_arousal);
    memory.participants = participants;
    memory.location = location;
    
    // Auto-calculate importance if not provided
    if (importance <= 0.0f) {
        memory.importance = CalculateImportance(description);
    } else {
        memory.importance = (importance < 0.0f) ? 0.0f : ((importance > 1.0f) ? 1.0f : importance);
    }
    
    memory.current_strength = 1.0f;  // Fresh memory
    memory.retrieval_count = 0;
    
    episodic_memories_.push_back(memory);
    
    return memory.event_id;
}

std::vector<EpisodicMemory> TemporalMemorySystem::RetrieveEpisodes(
    const std::string& query,
    int max_results,
    float min_strength
) {
    std::lock_guard<std::mutex> lock(mutex_);
    UpdateMemoryStrengths();
    
    // Score each memory by relevance * strength
    struct ScoredMemory {
        EpisodicMemory memory;
        float score;
    };
    
    std::vector<ScoredMemory> scored;
    scored.reserve(episodic_memories_.size());
    
    for (auto& memory : episodic_memories_) {
        if (memory.current_strength < min_strength) continue;
        
        float relevance = CalculateSimilarity(query, memory.description);
        float score = relevance * memory.current_strength;
        
        // Boost score for emotionally charged memories
        score *= (1.0f + memory.emotional_arousal * 0.5f);
        
        scored.push_back({memory, score});
        
        // Strengthen memory due to retrieval (spaced repetition effect)
        memory.current_strength = std::min(1.0f, memory.current_strength + retrieval_boost_);
        memory.retrieval_count++;
    }
    
    // Sort by score descending
    std::sort(scored.begin(), scored.end(), 
        [](const ScoredMemory& a, const ScoredMemory& b) {
            return a.score > b.score;
        });
    
    // Return top results
    std::vector<EpisodicMemory> results;
    int count = std::min(max_results, static_cast<int>(scored.size()));
    for (int i = 0; i < count; i++) {
        results.push_back(scored[i].memory);
    }
    
    return results;
}

std::vector<EpisodicMemory> TemporalMemorySystem::GetEpisodesWithEntity(
    const std::string& entity_name,
    int max_results
) {
    std::lock_guard<std::mutex> lock(mutex_);
    UpdateMemoryStrengths();
    
    std::vector<EpisodicMemory> results;
    
    for (const auto& memory : episodic_memories_) {
        // Check if entity is in participants
        bool found = std::find(memory.participants.begin(), 
                              memory.participants.end(), 
                              entity_name) != memory.participants.end();
        
        // Also check description
        if (!found) {
            found = memory.description.find(entity_name) != std::string::npos;
        }
        
        if (found && memory.current_strength > 0.1f) {
            results.push_back(memory);
        }
    }
    
    // Sort by strength * importance
    std::sort(results.begin(), results.end(),
        [](const EpisodicMemory& a, const EpisodicMemory& b) {
            return (a.current_strength * a.importance) > 
                   (b.current_strength * b.importance);
        });
    
    if (results.size() > static_cast<size_t>(max_results)) {
        results.resize(max_results);
    }
    
    return results;
}

std::string TemporalMemorySystem::AddSemanticKnowledge(
    const std::string& knowledge,
    float confidence,
    const std::string& source_episode
) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Check if similar knowledge already exists
    for (auto& semantic : semantic_memories_) {
        float similarity = CalculateSimilarity(knowledge, semantic.knowledge);
        if (similarity > 0.8f) {
            // Update existing knowledge
            semantic.confidence = std::max(semantic.confidence, confidence);
            semantic.last_updated = GetCurrentTimestamp();
            if (!source_episode.empty()) {
                semantic.source_episodes.push_back(source_episode);
            }
            return semantic.concept_id;
        }
    }
    
    // Create new semantic memory
    SemanticMemory semantic;
    semantic.concept_id = GenerateMemoryId();
    semantic.knowledge = knowledge;
    semantic.confidence = (confidence < 0.0f) ? 0.0f : ((confidence > 1.0f) ? 1.0f : confidence);
    semantic.first_learned = GetCurrentTimestamp();
    semantic.last_updated = semantic.first_learned;
    
    if (!source_episode.empty()) {
        semantic.source_episodes.push_back(source_episode);
    }
    
    semantic_memories_.push_back(semantic);
    
    return semantic.concept_id;
}

std::vector<SemanticMemory> TemporalMemorySystem::RetrieveSemanticKnowledge(
    const std::string& query,
    int max_results
) {
    std::lock_guard<std::mutex> lock(mutex_);
    struct ScoredSemantic {
        SemanticMemory memory;
        float score;
    };
    
    std::vector<ScoredSemantic> scored;
    
    for (const auto& semantic : semantic_memories_) {
        float relevance = CalculateSimilarity(query, semantic.knowledge);
        float score = relevance * semantic.confidence;
        scored.push_back({semantic, score});
    }
    
    std::sort(scored.begin(), scored.end(),
        [](const ScoredSemantic& a, const ScoredSemantic& b) {
            return a.score > b.score;
        });
    
    std::vector<SemanticMemory> results;
    int count = std::min(max_results, static_cast<int>(scored.size()));
    for (int i = 0; i < count; i++) {
        results.push_back(scored[i].memory);
    }
    
    return results;
}

void TemporalMemorySystem::ConsolidateMemories(
    std::function<std::string(const std::string&)> llm_callback
) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!llm_callback) return;
    
    // Find clusters of related episodes
    auto clusters = ClusterEpisodes();
    
    // Extract patterns from each cluster
    for (const auto& cluster_indices : clusters) {
        if (cluster_indices.size() < 3) continue;  // Need at least 3 episodes
        
        std::vector<EpisodicMemory> cluster;
        for (int idx : cluster_indices) {
            cluster.push_back(episodic_memories_[idx]);
        }
        
        // Use LLM to extract pattern
        std::string pattern = ExtractPattern(cluster, llm_callback);
        
        if (!pattern.empty()) {
            // Calculate confidence based on cluster size and episode importance
            float avg_importance = 0.0f;
            for (const auto& ep : cluster) {
                avg_importance += ep.importance;
            }
            avg_importance /= cluster.size();
            
            float confidence = std::min(0.9f, 0.5f + (cluster.size() * 0.05f) + avg_importance * 0.3f);
            
            // Add as semantic knowledge
            std::vector<std::string> source_ids;
            for (const auto& ep : cluster) {
                source_ids.push_back(ep.event_id);
            }
            
            AddSemanticKnowledge(pattern, confidence, source_ids[0]);
        }
    }
    
    // Prune very weak memories (< 0.05 strength)
    episodic_memories_.erase(
        std::remove_if(episodic_memories_.begin(), episodic_memories_.end(),
            [](const EpisodicMemory& m) { return m.current_strength < 0.05f; }),
        episodic_memories_.end()
    );
}

float TemporalMemorySystem::CalculateMemoryStrength(
    const EpisodicMemory& memory,
    int64_t current_time
) const {
    // Time elapsed in seconds
    int64_t elapsed = current_time - memory.timestamp;
    if (elapsed < 0) elapsed = 0;
    
    // Base decay using exponential forgetting curve
    float base_strength = std::exp(-decay_rate_ * elapsed);
    
    // Emotional memories decay slower
    float emotional_factor = 1.0f + (memory.emotional_arousal * emotional_boost_factor_);
    
    // Important memories decay slower
    float importance_factor = 1.0f + memory.importance;
    
    // Retrieval strengthens memory (spaced repetition)
    float retrieval_factor = 1.0f + (memory.retrieval_count * retrieval_boost_);
    
    float strength = base_strength * emotional_factor * importance_factor * retrieval_factor;
    
    return (strength < 0.0f) ? 0.0f : ((strength > 1.0f) ? 1.0f : strength);
}

void TemporalMemorySystem::UpdateMemoryStrengths() {
    // Note: This is a helper usually called from within a locked method.
    // If called directly, it should be locked.
    int64_t current_time = GetCurrentTimestamp();
    
    for (auto& memory : episodic_memories_) {
        memory.current_strength = CalculateMemoryStrength(memory, current_time);
    }
}


TemporalMemorySystem::MemoryStats TemporalMemorySystem::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryStats stats;
    stats.total_episodes = episodic_memories_.size();
    stats.total_semantic = semantic_memories_.size();
    stats.active_episodes = 0;
    stats.avg_episode_strength = 0.0f;
    stats.oldest_memory = INT64_MAX;
    stats.newest_memory = 0;
    
    for (const auto& ep : episodic_memories_) {
        if (ep.current_strength > 0.1f) {
            stats.active_episodes++;
        }
        stats.avg_episode_strength += ep.current_strength;
        stats.oldest_memory = std::min(stats.oldest_memory, ep.timestamp);
        stats.newest_memory = std::max(stats.newest_memory, ep.timestamp);
    }
    
    if (stats.total_episodes > 0) {
        stats.avg_episode_strength /= stats.total_episodes;
    }
    
    return stats;
}

// === Private Helper Functions ===

int64_t TemporalMemorySystem::GetCurrentTimestamp() const {
    return std::chrono::system_clock::now().time_since_epoch().count() / 1000000000;  // Seconds
}

float TemporalMemorySystem::CalculateImportance(const std::string& description) const {
    // Simple heuristic: longer descriptions and certain keywords indicate importance
    float importance = 0.3f;  // Base
    
    // Length factor
    importance += std::min(0.3f, description.length() / 500.0f);
    
    // Keyword boost
    std::vector<std::string> important_keywords = {
        "killed", "died", "betrayed", "saved", "discovered", "learned",
        "promised", "swore", "attacked", "defended", "loved", "hated"
    };
    
    for (const auto& keyword : important_keywords) {
        if (description.find(keyword) != std::string::npos) {
            importance += 0.1f;
        }
    }
    
    return (importance < 0.0f) ? 0.0f : ((importance > 1.0f) ? 1.0f : importance);
}

std::string TemporalMemorySystem::GenerateMemoryId() const {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    const char* hex_chars = "0123456789abcdef";
    std::string id = "mem_";
    for (int i = 0; i < 16; ++i) {
        id += hex_chars[dis(gen)];
    }
    return id;
}

float TemporalMemorySystem::CalculateSimilarity(const std::string& a, const std::string& b) const {
    // Simple word overlap similarity (can be replaced with embeddings)
    std::istringstream iss_a(a);
    std::istringstream iss_b(b);
    std::set<std::string> words_a, words_b;
    std::string word;
    
    while (iss_a >> word) words_a.insert(word);
    while (iss_b >> word) words_b.insert(word);
    
    if (words_a.empty() || words_b.empty()) return 0.0f;
    
    std::vector<std::string> intersection;
    std::set_intersection(words_a.begin(), words_a.end(),
                         words_b.begin(), words_b.end(),
                         std::back_inserter(intersection));
    
    float jaccard = static_cast<float>(intersection.size()) / 
                   (words_a.size() + words_b.size() - intersection.size());
    
    return jaccard;
}

std::vector<std::vector<int>> TemporalMemorySystem::ClusterEpisodes() const {
    // Simple clustering based on participant overlap and temporal proximity
    std::vector<std::vector<int>> clusters;
    std::vector<bool> assigned(episodic_memories_.size(), false);
    
    for (size_t i = 0; i < episodic_memories_.size(); i++) {
        if (assigned[i]) continue;
        
        std::vector<int> cluster;
        cluster.push_back(i);
        assigned[i] = true;
        
        // Find similar episodes
        for (size_t j = i + 1; j < episodic_memories_.size(); j++) {
            if (assigned[j]) continue;
            
            const auto& ep_i = episodic_memories_[i];
            const auto& ep_j = episodic_memories_[j];
            
            // Check participant overlap
            int common_participants = 0;
            for (const auto& p : ep_i.participants) {
                if (std::find(ep_j.participants.begin(), ep_j.participants.end(), p) != ep_j.participants.end()) {
                    common_participants++;
                }
            }
            
            // Check temporal proximity (within 1 day)
            int64_t time_diff = std::abs(ep_i.timestamp - ep_j.timestamp);
            bool temporally_close = time_diff < 86400;  // 1 day in seconds
            
            if (common_participants > 0 && temporally_close) {
                cluster.push_back(j);
                assigned[j] = true;
            }
        }
        
        if (cluster.size() >= 3) {
            clusters.push_back(cluster);
        }
    }
    
    return clusters;
}

std::string TemporalMemorySystem::ExtractPattern(
    const std::vector<EpisodicMemory>& cluster,
    std::function<std::string(const std::string&)> llm
) const {
    // Build prompt for LLM
    std::ostringstream prompt;
    prompt << "Extract a general pattern or belief from these related events:\n\n";
    
    for (const auto& ep : cluster) {
        prompt << "- " << ep.description << "\n";
    }
    
    prompt << "\nGeneral pattern or belief (one sentence):";
    
    return llm(prompt.str());
}

bool TemporalMemorySystem::Save(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        nlohmann::json j;
        
        j["episodic_memories"] = nlohmann::json::array();
        for (const auto& mem : episodic_memories_) {
            nlohmann::json m;
            m["event_id"] = mem.event_id;
            m["description"] = mem.description;
            m["timestamp"] = mem.timestamp;
            m["emotional_valence"] = mem.emotional_valence;
            m["emotional_arousal"] = mem.emotional_arousal;
            m["importance"] = mem.importance;
            m["participants"] = mem.participants;
            m["location"] = mem.location;
            m["current_strength"] = mem.current_strength;
            m["retrieval_count"] = mem.retrieval_count;
            j["episodic_memories"].push_back(m);
        }
        
        j["semantic_memories"] = nlohmann::json::array();
        for (const auto& mem : semantic_memories_) {
            nlohmann::json m;
            m["concept_id"] = mem.concept_id;
            m["knowledge"] = mem.knowledge;
            m["confidence"] = mem.confidence;
            m["source_episodes"] = mem.source_episodes;
            m["first_learned"] = mem.first_learned;
            m["last_updated"] = mem.last_updated;
            j["semantic_memories"].push_back(m);
        }
        
        std::ofstream file(filepath);
        if (!file.is_open()) return false;
        file << std::setw(4) << j << std::endl;
        return true;
    } catch (const std::exception& e) {
        NPCLogger::Error(std::string("Error saving TemporalMemorySystem: ") + e.what());
        return false;
    }
}

bool TemporalMemorySystem::Load(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        
        nlohmann::json j;
        file >> j;
        
        episodic_memories_.clear();
        if (j.contains("episodic_memories") && j["episodic_memories"].is_array()) {
            for (const auto& item : j["episodic_memories"]) {
                EpisodicMemory mem;
                mem.event_id = item.value("event_id", "");
                mem.description = item.value("description", "");
                mem.timestamp = item.value("timestamp", 0LL);
                mem.emotional_valence = item.value("emotional_valence", 0.0f);
                mem.emotional_arousal = item.value("emotional_arousal", 0.0f);
                mem.importance = item.value("importance", 0.0f);
                if (item.contains("participants") && item["participants"].is_array()) {
                    mem.participants = item["participants"].get<std::vector<std::string>>();
                }
                mem.location = item.value("location", "");
                mem.current_strength = item.value("current_strength", 1.0f);
                mem.retrieval_count = item.value("retrieval_count", 0);
                episodic_memories_.push_back(mem);
            }
        }
        
        semantic_memories_.clear();
        if (j.contains("semantic_memories") && j["semantic_memories"].is_array()) {
            for (const auto& item : j["semantic_memories"]) {
                SemanticMemory mem;
                mem.concept_id = item.value("concept_id", "");
                mem.knowledge = item.value("knowledge", "");
                mem.confidence = item.value("confidence", 0.0f);
                if (item.contains("source_episodes") && item["source_episodes"].is_array()) {
                    mem.source_episodes = item["source_episodes"].get<std::vector<std::string>>();
                }
                mem.first_learned = item.value("first_learned", 0LL);
                mem.last_updated = item.value("last_updated", 0LL);
                semantic_memories_.push_back(mem);
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        NPCLogger::Error(std::string("Error loading TemporalMemorySystem: ") + e.what());
        return false;
    }
}

} // namespace NPCInference
