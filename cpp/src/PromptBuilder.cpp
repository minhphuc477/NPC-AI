#include "PromptBuilder.h"
#include <iostream>
#include <sstream>
#include <map>

namespace NPCInference {

    PromptBuilder::PromptBuilder(bool useAdvancedFormat, bool useJsonFormat) : useAdvancedFormat_(useAdvancedFormat), useJsonFormat_(useJsonFormat) {}

    std::string PromptBuilder::Build(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& language, const json& tools) {
        if (useAdvancedFormat_) {
            return BuildAdvanced(npcData, gameState, playerInput, language, tools);
        }
        else {
            return BuildLegacy(npcData, gameState, playerInput);
        }
    }

    std::string PromptBuilder::BuildAdvanced(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& language, const json& tools) {
        bool isVi = (language == "vi");

        // --- Localized Labels ---
        const char* L_STATE    = isVi ? "**Trạng thái hiện tại:**" : "**Current State:**";
        const char* L_BEHAVIOR = isVi ? "- Hành vi: " : "- Behavior: ";
        const char* L_MOOD     = isVi ? "- Tâm trạng: " : "- Mood: ";
        const char* L_HEALTH   = isVi ? "- Sức khỏe: " : "- Health: ";
        const char* L_EMOTION  = isVi ? "- Cảm xúc: " : "- Emotional State: ";
        
        const char* L_ENV      = isVi ? "**Môi trường:**" : "**Environment:**";
        const char* L_LOC      = isVi ? "- Địa điểm: " : "- Location: ";
        const char* L_TIME     = isVi ? "- Thời gian: " : "- Time: ";
        const char* L_NEARBY   = isVi ? "- Xung quanh: " : "- Nearby: ";
        
        const char* L_RELATION = isVi ? "**Mối quan hệ:**" : "**Relationship with Player:**";
        const char* L_TRUST    = isVi ? "- Tin tưởng: " : "- Trust: ";
        
        const char* L_HISTORY_HEADER = isVi ? "**Lịch sử trò chuyện:**" : "**Conversation History:**";
        const char* L_SCENARIO       = isVi ? "**Bối cảnh:**" : "**Scenario:**";

        // --- Data Extraction ---
        std::string name = npcData.value("name", "NPC");
        std::string personaKey = isVi ? "persona_vi" : "persona_en";
        std::string persona = npcData.value(personaKey, npcData.value("persona", isVi ? "Bạn là một NPC hữu ích." : "You are a helpful NPC."));

        std::string behavior = gameState.value("behavior_state", "Idle");
        std::string mood = gameState.value("mood_state", "Neutral");
        std::string health = gameState.value("health_state", "Healthy");

        std::string location = gameState.value("location", "Unknown");
        std::string timeOfDay = gameState.value("time_of_day", "Daytime");
        std::string nearby = gameState.value("nearby_entities", "None");

        int trust = gameState.value("trust_level", 50);
        std::string scenario = gameState.value("scenario_plot", "No specific scenario.");
        
        std::string memory_context = gameState.value("memory_context", "");

        std::stringstream ss;
        
        // System Prompt Start
        ss << "<|system|>\n";
        
        if (isVi) {
            ss << "Bạn là " << name << ". " << persona << "\n\n";
        } else {
            ss << "You are " << name << ". " << persona << "\n\n";
        }

        // Tools Injection
        if (!tools.is_null() && !tools.empty()) {
            if (isVi) {
                ss << "Bạn có thể sử dụng các công cụ sau:\n";
                ss << tools.dump(4) << "\n\n";
                ss << "Để sử dụng công cụ, hãy trả về JSON:\n";
                ss << "{\"tool\": \"tool_name\", \"args\": {...}}\n\n"; 
            } else {
                ss << "You have access to the following tools:\n";
                ss << tools.dump(4) << "\n\n";
                ss << "To use a tool, output a JSON object:\n";
                ss << "{\"tool\": \"tool_name\", \"args\": {...}}\n\n";
            }
        }

        // State Block
        ss << L_STATE << "\n"
           << L_BEHAVIOR << behavior << "\n"
           << L_MOOD << mood << "\n"
           << L_HEALTH << health << "\n";

        // Emotional State (if present)
        if (gameState.contains("emotional_state") && gameState["emotional_state"].is_object()) {
            // Find dominant emotion
            std::string dominant = "neutral";
            double maxVal = -1.0;
            json emo = gameState["emotional_state"];
            for (auto& el : emo.items()) {
                if (el.value().is_number() && (double)el.value() > maxVal) {
                    maxVal = (double)el.value();
                    dominant = el.key();
                }
            }
            if (maxVal > 0) {
                ss << L_EMOTION << dominant << " (" << (int)(maxVal * 100) << "%)\n";
            }
        }
        ss << "\n";

        // Environment Block
        ss << L_ENV << "\n"
           << L_LOC << location << "\n"
           << L_TIME << timeOfDay << "\n"
           << L_NEARBY << nearby << "\n\n";

        // Relationship Block
        ss << L_RELATION << "\n"
           << L_TRUST << trust << "/100\n";
           
        // Memory Block (if exists)
        if (!memory_context.empty()) {
            ss << "\n" << L_HISTORY_HEADER << "\n" << memory_context << "\n";
        }

        // Scenario Block
        ss << "\n" << L_SCENARIO << "\n"
           << scenario << "\n";

        // Chat Template Format
        ss << "<|end|>\n"
           << "<|user|>\n"
           << playerInput << "\n"
           << "<|assistant|>\n";


        // JSON Output Instruction (Functional Grounding)
        if (useJsonFormat_) {
            ss << "\n";
            if (isVi) {
                ss << "QUAN TRỌNG: Bạn phải trả lời bằng format JSON. Không trả lời bằng text thường.\n";
                ss << "Schema:\n{\n";
                ss << "  \"text\": \"Câu trả lời của bạn (lời thoại)\",\n";
                ss << "  \"emotion\": \"Cảm xúc hiện tại (Vui/Buồn/Giận/Sợ/Ngạc nhiên)\",\n";
                ss << "  \"action\": \"Hành động (Ví dụ: Node/Cuoi/Rút kiếm/Bỏ đi)\",\n";
                ss << "  \"trust_change\": \"Thay đổi mức tin tưởng (-10 đến +10)\"\n";
                ss << "}\n";
            } else {
                ss << "IMPORTANT: You must respond in JSON format. Do not use plain text.\n";
                ss << "Include a hidden 'thought' field to analyze the situation before speaking.\n";
                ss << "Schema:\n{\n";
                ss << "  \"thought\": \"Internal monologue/reasoning about the situation\",\n";
                ss << "  \"text\": \"Your dialogue response\",\n";
                ss << "  \"emotion\": \"Current emotion (Joy/Sadness/Anger/Fear/Surprise)\",\n";
                ss << "  \"action\": \"Action to perform (e.g., Nod/Smile/Draw_Sword/Walk_Away)\",\n";
                ss << "  \"trust_change\": \"Trust level change (-10 to +10)\"\n";
                ss << "}\n";
            }
        }

        return ss.str();
    }

    std::string PromptBuilder::BuildLegacy(const json& npcData, const json& gameState, const std::string& playerInput) {
        std::string persona = npcData.value("persona", "You are a helpful NPC.");
        std::string npcId = npcData.value("id", "NPC");
        std::string scenario = gameState.value("scenario", "");

        return "System: " + persona + "\nName: " + npcId + "\nContext: " + scenario + "\n\nQuestion: " + playerInput + "\nAnswer:";
    }

} // namespace NPCInference
