#include "PlayerBehaviorModeling.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

namespace NPCInference {

PlayerBehaviorModeling::PlayerBehaviorModeling() {
    // Initialize with neutral profile
    profile_ = PlayerProfile();
}

std::string PlayerBehaviorModeling::RecordAction(
    const std::string& action_type,
    const std::string& target,
    const std::string& context,
    bool was_successful,
    float risk_level
) {
    std::lock_guard<std::mutex> lock(mutex_);
    PlayerAction action;
    action.action_type = action_type;
    action.target = target;
    action.context = context;
    action.timestamp = GetCurrentTimestamp();
    action.was_successful = was_successful;
    action.risk_level = (risk_level < 0.0f) ? 0.0f : ((risk_level > 1.0f) ? 1.0f : risk_level);
    
    // Add to history (maintain max size)
    action_history_.push_back(action);
    if (action_history_.size() > static_cast<size_t>(max_history_size_)) {
        action_history_.pop_front();
    }
    
    // Update profile incrementally
    UpdateProfile();
    
    // Detect patterns periodically
    if (action_history_.size() % pattern_detection_window_ == 0) {
        DetectPatterns();
    }
    
    return GenerateActionId();
}

std::vector<PlayerAction> PlayerBehaviorModeling::GetRecentActions(int count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<PlayerAction> recent;
    int start = std::max(0, static_cast<int>(action_history_.size()) - count);
    for (size_t i = start; i < action_history_.size(); i++) {
        recent.push_back(action_history_[i]);
    }
    return recent;
}

void PlayerBehaviorModeling::DetectPatterns() {
    // Note: This is private and called by RecordAction which already holds the lock.
    // If called from public methods, it would need a recursive mutex or a private _NoLock version.
    // I'll add logic here directly.
    if (action_history_.size() < 5) return;
    
    // Clear old patterns with low confidence
    detected_patterns_.erase(
        std::remove_if(detected_patterns_.begin(), detected_patterns_.end(),
            [](const BehaviorPattern& p) { return p.confidence < 0.3f; }),
        detected_patterns_.end()
    );
    
    // === Pattern 1: Aggression ===
    int aggressive_actions = 0;
    int total_combat = 0;
    for (const auto& action : action_history_) {
        if (action.action_type == "attack" || action.action_type == "charge" || 
            action.action_type == "ambush") {
            aggressive_actions++;
            total_combat++;
        } else if (action.action_type == "defend" || action.action_type == "retreat") {
            total_combat++;
        }
    }
    
    if (total_combat > 0) {
        float aggression_ratio = static_cast<float>(aggressive_actions) / total_combat;
        if (aggression_ratio > 0.7f) {
            BehaviorPattern pattern;
            pattern.pattern_type = "aggressive";
            pattern.description = "Player prefers offensive tactics";
            pattern.confidence = aggression_ratio;
            pattern.occurrence_count = aggressive_actions;
            pattern.first_detected = GetCurrentTimestamp();
            pattern.last_seen = GetCurrentTimestamp();
            detected_patterns_.push_back(pattern);
        }
    }
}

std::vector<BehaviorPattern> PlayerBehaviorModeling::GetPatterns(float min_confidence) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<BehaviorPattern> filtered;
    for (const auto& pattern : detected_patterns_) {
        if (pattern.confidence >= min_confidence) {
            filtered.push_back(pattern);
        }
    }
    return filtered;
}

float PlayerBehaviorModeling::HasPattern(const std::string& pattern_type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& pattern : detected_patterns_) {
        if (pattern.pattern_type == pattern_type) {
            return pattern.confidence;
        }
    }
    return 0.0f;
}

void PlayerBehaviorModeling::UpdateProfile() {
    // Note: This is private and called by methods that hold the lock
    if (action_history_.empty()) return;
    
    UpdatePlaystyleDimensions();
    UpdateSkillAssessment();
    
    // Update preferred/avoided actions
    profile_.preferred_actions.clear();
    profile_.avoided_actions.clear();
    
    std::map<std::string, int> action_counts;
    for (const auto& action : action_history_) {
        action_counts[action.action_type]++;
    }
    
    for (const auto& [action_type, count] : action_counts) {
        float frequency = static_cast<float>(count) / action_history_.size();
        if (frequency > 0.2f) {
            profile_.preferred_actions[action_type] = frequency;
        } else if (frequency < 0.05f) {
            profile_.avoided_actions[action_type] = frequency;
        }
    }
}

std::vector<std::pair<std::string, float>> PlayerBehaviorModeling::PredictNextAction(
    const std::string& current_context,
    int top_n
) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (action_history_.empty()) return {};
    
    // Build probability distribution based on historical frequency
    std::map<std::string, float> action_probs;
    
    // Count actions in similar contexts
    std::map<std::string, int> context_action_counts;
    int total_context_actions = 0;
    
    for (const auto& action : action_history_) {
        if (action.context == current_context || current_context.empty()) {
            context_action_counts[action.action_type]++;
            total_context_actions++;
        }
    }
    
    // If no context matches, use overall distribution
    if (total_context_actions == 0) {
        for (const auto& action : action_history_) {
            context_action_counts[action.action_type]++;
            total_context_actions++;
        }
    }
    
    // Calculate probabilities
    for (const auto& [action_type, count] : context_action_counts) {
        action_probs[action_type] = static_cast<float>(count) / total_context_actions;
    }
    
    // Sort by probability
    std::vector<std::pair<std::string, float>> predictions(action_probs.begin(), action_probs.end());
    std::sort(predictions.begin(), predictions.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Return top N
    if (predictions.size() > static_cast<size_t>(top_n)) {
        predictions.resize(top_n);
    }
    
    return predictions;
}

float PlayerBehaviorModeling::EstimateActionProbability(
    const std::string& action_type,
    const std::string& context
) const {
    // PredictNextAction handles its own locking
    auto predictions = PredictNextAction(context, 10);
    for (const auto& [type, prob] : predictions) {
        if (type == action_type) {
            return prob;
        }
    }
    return 0.0f;
}

std::string PlayerBehaviorModeling::SuggestCounterStrategy(const std::string& npc_goal) const {
    std::lock_guard<std::mutex> lock(mutex_);
    // Analyze player profile and suggest counter
    std::stringstream strategy;
    
    if (profile_.aggression > 0.7f) {
        strategy << "Player is aggressive. Use defensive tactics and counterattacks.";
    } else if (profile_.caution > 0.7f) {
        strategy << "Player is cautious. Apply pressure to force mistakes.";
    }
    
    if (profile_.social_preference > 0.6f) {
        strategy << " Player prefers dialogue. Prepare strong arguments.";
    } else if (profile_.social_preference < 0.4f) {
        strategy << " Player prefers combat. Strengthen defenses.";
    }
    
    // Check for repetitive patterns
    for (const auto& pattern : detected_patterns_) {
        if (pattern.pattern_type.find("repetitive_") == 0 && pattern.confidence > 0.6f) {
            std::string overused = pattern.pattern_type.substr(11);  // Remove "repetitive_"
            strategy << " Player overuses " << overused << ". Prepare specific counter.";
        }
    }
    
    if (profile_.creativity < 0.3f) {
        strategy << " Player is predictable. Exploit patterns.";
    } else if (profile_.creativity > 0.7f) {
        strategy << " Player is unpredictable. Stay adaptable.";
    }
    
    return strategy.str();
}

float PlayerBehaviorModeling::AssessPredictability() const {
    std::lock_guard<std::mutex> lock(mutex_);
    // Low entropy = high predictability
    return 1.0f - profile_.creativity;
}

PlayerBehaviorModeling::ModelingStats PlayerBehaviorModeling::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ModelingStats stats;
    stats.total_actions = static_cast<int>(action_history_.size());
    stats.patterns_detected = static_cast<int>(detected_patterns_.size());
    
    int successes = 0;
    for (const auto& action : action_history_) {
        if (action.was_successful) successes++;
    }
    stats.avg_success_rate = action_history_.empty() ? 0.0f : 
        static_cast<float>(successes) / action_history_.size();
    
    stats.profile_confidence = action_history_.size() >= 20 ? 0.8f : 
        static_cast<float>(action_history_.size()) / 20.0f;
    
    // Determine dominant playstyle
    if (profile_.aggression > 0.7f) {
        stats.dominant_playstyle = "Aggressive";
    } else if (profile_.caution > 0.7f) {
        stats.dominant_playstyle = "Cautious";
    } else if (profile_.social_preference > 0.6f) {
        stats.dominant_playstyle = "Diplomatic";
    } else {
        stats.dominant_playstyle = "Balanced";
    }
    
    return stats;
}

bool PlayerBehaviorModeling::Save(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        nlohmann::json j = ToJSON();
        std::ofstream file(filepath);
        if (!file.is_open()) return false;
        file << std::setw(2) << j;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving PlayerBehaviorModeling: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error occurred saving PlayerBehaviorModeling" << std::endl;
        return false;
    }
}

bool PlayerBehaviorModeling::Load(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        
        // RAM optimization: Streaming JSON load
        nlohmann::json j;
        try {
            file >> j;
        } catch (const nlohmann::json::parse_error& e) {
            std::cerr << "JSON Parse Error in PlayerBehaviorModeling: " << e.what() << std::endl;
            return false;
        }

        FromJSON(j);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading PlayerBehaviorModeling: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error occurred loading PlayerBehaviorModeling" << std::endl;
        return false;
    }
}

nlohmann::json PlayerBehaviorModeling::ToJSON() const {
    // No lock here as it's a private helper called by Save() which locks
    nlohmann::json j;
    
    // Save profile
    j["profile"]["aggression"] = profile_.aggression;
    j["profile"]["caution"] = profile_.caution;
    j["profile"]["social_preference"] = profile_.social_preference;
    j["profile"]["exploration_tendency"] = profile_.exploration_tendency;
    j["profile"]["creativity"] = profile_.creativity;
    j["profile"]["estimated_skill"] = profile_.estimated_skill;
    j["profile"]["reaction_speed"] = profile_.reaction_speed;
    j["profile"]["strategic_thinking"] = profile_.strategic_thinking;
    
    // Save patterns
    j["patterns"] = nlohmann::json::array();
    for (const auto& pattern : detected_patterns_) {
        nlohmann::json p;
        p["type"] = pattern.pattern_type;
        p["description"] = pattern.description;
        p["confidence"] = pattern.confidence;
        p["count"] = pattern.occurrence_count;
        j["patterns"].push_back(p);
    }
    
    // Save recent actions (last 20)
    j["recent_actions"] = nlohmann::json::array();
    int start = std::max(0, static_cast<int>(action_history_.size()) - 20);
    for (size_t i = start; i < action_history_.size(); i++) {
        nlohmann::json a;
        a["type"] = action_history_[i].action_type;
        a["target"] = action_history_[i].target;
        a["success"] = action_history_[i].was_successful;
        a["risk"] = action_history_[i].risk_level;
        j["recent_actions"].push_back(a);
    }
    
    return j;
}

void PlayerBehaviorModeling::FromJSON(const nlohmann::json& j) {
    // No lock here as it's a private helper called by Load() which locks
    if (j.contains("profile")) {
        profile_.aggression = j["profile"].value("aggression", 0.5f);
        profile_.caution = j["profile"].value("caution", 0.5f);
        profile_.social_preference = j["profile"].value("social_preference", 0.5f);
        profile_.exploration_tendency = j["profile"].value("exploration_tendency", 0.5f);
        profile_.creativity = j["profile"].value("creativity", 0.5f);
        profile_.estimated_skill = j["profile"].value("estimated_skill", 0.5f);
        profile_.reaction_speed = j["profile"].value("reaction_speed", 0.5f);
        profile_.strategic_thinking = j["profile"].value("strategic_thinking", 0.5f);
    }
    
    // Load patterns
    if (j.contains("patterns")) {
        detected_patterns_.clear();
        for (const auto& p : j["patterns"]) {
            BehaviorPattern pattern;
            pattern.pattern_type = p.value("type", "");
            pattern.description = p.value("description", "");
            pattern.confidence = p.value("confidence", 0.5f);
            pattern.occurrence_count = p.value("count", 0);
            detected_patterns_.push_back(pattern);
        }
    }
}

int64_t PlayerBehaviorModeling::GetCurrentTimestamp() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

std::string PlayerBehaviorModeling::GenerateActionId() const {
    std::stringstream ss;
    ss << "action_" << GetCurrentTimestamp() << "_" << action_history_.size();
    return ss.str();
}

float PlayerBehaviorModeling::CalculateActionSimilarity(
    const PlayerAction& a,
    const PlayerAction& b
) const {
    float score = 0.0f;
    if (a.action_type == b.action_type) score += 0.4f;
    if (!a.target.empty() && a.target == b.target) score += 0.2f;
    if (!a.context.empty() && a.context == b.context) score += 0.2f;
    if (a.was_successful == b.was_successful) score += 0.1f;
    score += (1.0f - std::min(1.0f, std::abs(a.risk_level - b.risk_level))) * 0.1f;
    return std::max(0.0f, std::min(1.0f, score));
}

std::vector<std::vector<int>> PlayerBehaviorModeling::ClusterActions() const {
    std::vector<std::vector<int>> clusters;
    if (action_history_.empty()) return clusters;

    std::map<std::string, std::vector<int>> by_type;
    for (int i = 0; i < static_cast<int>(action_history_.size()); ++i) {
        by_type[action_history_[i].action_type].push_back(i);
    }

    for (auto& [_, indices] : by_type) {
        if (!indices.empty()) {
            clusters.push_back(indices);
        }
    }
    return clusters;
}

BehaviorPattern PlayerBehaviorModeling::AnalyzeCluster(const std::vector<PlayerAction>& cluster) const {
    BehaviorPattern pattern;
    if (cluster.empty()) return pattern;

    std::map<std::string, int> counts;
    for (const auto& action : cluster) {
        counts[action.action_type]++;
    }

    std::string dominant = cluster.front().action_type;
    int max_count = 0;
    for (const auto& [action_type, count] : counts) {
        if (count > max_count) {
            dominant = action_type;
            max_count = count;
        }
    }

    pattern.pattern_type = "cluster_" + dominant;
    pattern.description = "Repeated preference for action type: " + dominant;
    pattern.occurrence_count = max_count;
    pattern.confidence = static_cast<float>(max_count) / static_cast<float>(cluster.size());
    pattern.first_detected = cluster.front().timestamp;
    pattern.last_seen = cluster.back().timestamp;
    return pattern;
}

void PlayerBehaviorModeling::UpdatePlaystyleDimensions() {
    if (action_history_.empty()) return;

    const std::set<std::string> aggressive_actions = {"attack", "charge", "ambush", "intimidate"};
    const std::set<std::string> cautious_actions = {"defend", "retreat", "hide", "observe"};
    const std::set<std::string> social_actions = {"talk", "negotiate", "persuade", "trade"};
    const std::set<std::string> exploration_actions = {"explore", "investigate", "search", "scout"};

    float aggressive = 0.0f;
    float cautious = 0.0f;
    float social = 0.0f;
    float exploration = 0.0f;
    float risk_sum = 0.0f;

    for (const auto& action : action_history_) {
        risk_sum += action.risk_level;
        if (aggressive_actions.count(action.action_type)) aggressive += 1.0f;
        if (cautious_actions.count(action.action_type)) cautious += 1.0f;
        if (social_actions.count(action.action_type)) social += 1.0f;
        if (exploration_actions.count(action.action_type)) exploration += 1.0f;
    }

    const float total = static_cast<float>(action_history_.size());
    const float avg_risk = risk_sum / std::max(1.0f, total);
    profile_.aggression = std::max(0.0f, std::min(1.0f, (aggressive / total + avg_risk) * 0.5f));
    profile_.caution = std::max(0.0f, std::min(1.0f, (cautious / total + (1.0f - avg_risk)) * 0.5f));
    profile_.social_preference = std::max(0.0f, std::min(1.0f, social / total));
    profile_.exploration_tendency = std::max(0.0f, std::min(1.0f, exploration / total));
    profile_.creativity = std::max(0.0f, std::min(1.0f, CalculateEntropy()));
}

void PlayerBehaviorModeling::UpdateSkillAssessment() {
    if (action_history_.empty()) return;

    float success_count = 0.0f;
    float risk_success = 0.0f;
    int64_t total_delta = 0;
    int delta_count = 0;

    for (size_t i = 0; i < action_history_.size(); ++i) {
        const auto& action = action_history_[i];
        if (action.was_successful) {
            success_count += 1.0f;
            risk_success += action.risk_level;
        }
        if (i > 0) {
            int64_t dt = action.timestamp - action_history_[i - 1].timestamp;
            if (dt >= 0) {
                total_delta += dt;
                delta_count++;
            }
        }
    }

    const float total = static_cast<float>(action_history_.size());
    const float success_rate = success_count / std::max(1.0f, total);
    const float avg_success_risk = success_count > 0 ? risk_success / success_count : 0.0f;
    const float avg_delta = delta_count > 0 ? static_cast<float>(total_delta) / delta_count : 30.0f;
    const float reaction = 1.0f / (1.0f + avg_delta / 20.0f);

    profile_.reaction_speed = std::max(0.0f, std::min(1.0f, reaction));
    profile_.strategic_thinking = std::max(0.0f, std::min(1.0f, 0.6f * success_rate + 0.4f * avg_success_risk));
    profile_.estimated_skill = std::max(
        0.0f,
        std::min(1.0f, 0.5f * success_rate + 0.3f * profile_.strategic_thinking + 0.2f * profile_.reaction_speed)
    );
}

float PlayerBehaviorModeling::CalculateEntropy() const {
    if (action_history_.empty()) return 0.0f;

    std::map<std::string, int> counts;
    for (const auto& action : action_history_) {
        counts[action.action_type]++;
    }

    const float n = static_cast<float>(action_history_.size());
    if (n <= 0.0f) return 0.0f;

    float entropy = 0.0f;
    for (const auto& [_, count] : counts) {
        const float p = static_cast<float>(count) / n;
        if (p > 1e-8f) {
            entropy -= p * std::log2(p);
        }
    }

    const float max_entropy = counts.size() > 1 ? std::log2(static_cast<float>(counts.size())) : 1.0f;
    return entropy / max_entropy;
}

} // namespace NPCInference
