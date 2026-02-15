#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <chrono>
#include <random>

namespace NPCInference {

struct Message {
    std::string role;  // "user" or "assistant"
    std::string content;
    int64_t timestamp;
    
    Message(const std::string& r, const std::string& c) 
        : role(r), content(c) {
        timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    }
};

struct ConversationContext {
    std::string session_id;
    std::string npc_name;
    std::string player_name;
    std::vector<Message> history;
    std::string location;
    std::string time_of_day;
    std::map<std::string, std::string> metadata;
    int64_t created_at;
    int64_t last_active;
    
    ConversationContext() {
        created_at = std::chrono::system_clock::now().time_since_epoch().count();
        last_active = created_at;
    }
};

class ConversationManager {
private:
    std::map<std::string, ConversationContext> active_sessions_;
    std::mutex sessions_mutex_;
    
    std::string GenerateSessionId() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, 15);
        
        const char* hex_chars = "0123456789abcdef";
        std::string id;
        for (int i = 0; i < 16; ++i) {
            id += hex_chars[dis(gen)];
        }
        return id;
    }
    
public:
    std::string CreateSession(const std::string& npc_name, const std::string& player_name) {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        
        ConversationContext ctx;
        ctx.session_id = GenerateSessionId();
        ctx.npc_name = npc_name;
        ctx.player_name = player_name;
        
        active_sessions_[ctx.session_id] = ctx;
        return ctx.session_id;
    }
    
    ConversationContext* GetSession(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        
        auto it = active_sessions_.find(session_id);
        if (it != active_sessions_.end()) {
            it->second.last_active = std::chrono::system_clock::now().time_since_epoch().count();
            return &(it->second);
        }
        return nullptr;
    }
    
    void AddMessage(const std::string& session_id, const std::string& role, const std::string& content) {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        
        auto it = active_sessions_.find(session_id);
        if (it != active_sessions_.end()) {
            it->second.history.emplace_back(role, content);
            it->second.last_active = std::chrono::system_clock::now().time_since_epoch().count();
        }
    }
    
    std::vector<Message> GetHistory(const std::string& session_id, int max_messages = 10) {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        
        auto it = active_sessions_.find(session_id);
        if (it != active_sessions_.end()) {
            const auto& history = it->second.history;
            if (history.size() <= static_cast<size_t>(max_messages)) {
                return history;
            }
            
            // Return last N messages
            return std::vector<Message>(
                history.end() - max_messages,
                history.end()
            );
        }
        return {};
    }
    
    void CloseSession(const std::string& session_id) {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        active_sessions_.erase(session_id);
    }
    
    size_t GetActiveSessionCount() {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        return active_sessions_.size();
    }
};

} // namespace NPCInference
