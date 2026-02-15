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

    }

    std::string PromptBuilder::BuildPlanning(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& language) {
        bool isVi = (language == "vi");
        std::string basePrompt = Build(npcData, gameState, playerInput, language);
        
        std::string marker = "<|assistant|>\n";
        size_t pos = basePrompt.find(marker);
        std::string prefix = (pos != std::string::npos) ? basePrompt.substr(0, pos) : basePrompt;

        if (isVi) {
            return prefix + marker + "**Suy nghĩ:** Hãy phân tích tình huống và lập kế hoạch câu trả lời.\n";
        } else {
            return prefix + marker + "**Thought:** Analyze the situation and plan your response.\n";
        }
    }

    std::string PromptBuilder::BuildWithThought(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& thought, const std::string& language, const json& tools) {
        bool isVi = (language == "vi");
        std::string basePrompt = Build(npcData, gameState, playerInput, language, tools);
        
        std::string marker = "<|assistant|>\n";
        size_t pos = basePrompt.find(marker);
        if (pos != std::string::npos) {
            std::string label = isVi ? "**Suy nghĩ:** " : "**Thought:** ";
            basePrompt.insert(pos + marker.length(), label + thought + "\n\n");
        }
        
        return basePrompt;
    }

    std::string PromptBuilder::BuildCritique(const std::string& originalResponse, const json& npcData, const json& gameState, const std::string& language) {
        bool isVi = (language == "vi");
        std::stringstream ss;
        
        if (isVi) {
            ss << "<|system|>\nBạn là một nhà phê bình nghiêm túc. Hãy đánh giá câu trả lời sau đây của NPC " << npcData.value("name", "NPC") 
               << " dựa trên: 1. Tính nhất quán của persona, 2. Độ chính xác của thông tin, 3. Sự tự nhiên của ngôn ngữ.\n\n"
               << "Câu trả lời của NPC:\n\"" << originalResponse << "\"\n\n"
               << "Hãy chỉ ra các lỗi hoặc điểm cần cải thiện. Nếu câu trả lời đã hoàn hảo, hãy trả về 'PERFECT'.\n"
               << "<|end|>\n<|assistant|>\n**Phê bình:** ";
        } else {
            ss << "<|system|>\nYou are a critical reviewer. Evaluate the following response from NPC " << npcData.value("name", "NPC")
               << " based on: 1. Persona consistency, 2. Fact accuracy, 3. Naturalness.\n\n"
               << "NPC Response:\n\"" << originalResponse << "\"\n\n"
               << "Point out flaws or areas for improvement. If the response is perfect, return 'PERFECT'.\n"
               << "<|end|>\n<|assistant|>\n**Critique:** ";
        }
        
        return ss.str();
    }

    std::string PromptBuilder::BuildRefine(const std::string& originalResponse, const std::string& critique, const json& npcData, const json& gameState, const std::string& language) {
        bool isVi = (language == "vi");
        std::stringstream ss;
        
        if (isVi) {
            ss << "<|system|>\nBạn là " << npcData.value("name", "NPC") << ". Dựa trên những nhận xét phê bình sau đây, hãy viết lại câu trả lời của bạn để nó hoàn thiện hơn.\n\n"
               << "Nhận xét phê bình:\n" << critique << "\n\n"
               << "Câu trả lời cũ:\n\"" << originalResponse << "\"\n\n"
               << "Hãy đưa ra câu trả lời mới tinh tế và nhất quán hơn.\n"
               << "<|end|>\n<|assistant|>\n";
        } else {
            ss << "<|system|>\nYou are " << npcData.value("name", "NPC") << ". Based on the following critique, rewrite your response to improve it.\n\n"
               << "Critique:\n" << critique << "\n\n"
               << "Old Response:\n\"" << originalResponse << "\"\n\n"
               << "Provide a newer, more refined and consistent response.\n"
               << "<|end|>\n<|assistant|>\n";
        }
        
        return ss.str();
    }

    std::string PromptBuilder::BuildTruthGuardCheck(const std::string& response, const std::string& worldFacts, const std::string& language) {
        bool isVi = (language == "vi");
        std::stringstream ss;
        
        if (isVi) {
            ss << "<|system|>\nBạn là một Hệ thống Kiểm chứng Sự thật (Truth Guard). Hãy so sánh câu trả lời của NPC với các Sự thật Thế giới sau đây.\n\n"
               << "Sự thật Thế giới (Dữ liệu gốc):\n" << worldFacts << "\n\n"
               << "Câu trả lời của NPC:\n\"" << response << "\"\n\n"
               << "Nếu câu trả lời mâu thuẫn với Sự thật Thế giới, hãy liệt kê các mâu thuẫn. Nếu không có mâu thuẫn, hãy trả về 'VALID'.\n"
               << "<|end|>\n<|assistant|>\n**Kiểm chứng:** ";
        } else {
            ss << "<|system|>\nYou are a Truth Guard system. Compare the NPC's response with the following World Facts.\n\n"
               << "World Facts (Symbolic Ground Truth):\n" << worldFacts << "\n\n"
               << "NPC Response:\n\"" << response << "\"\n\n"
               << "If the response contradicts World Facts, list the contradictions. If there are no contradictions, return 'VALID'.\n"
               << "<|end|>\n<|assistant|>\n**Verification:** ";
        }
        
        return ss.str();
    }

    std::string PromptBuilder::BuildOIEPrompt(const std::string& text, const std::string& language) {
        bool isVi = (language == "vi");
        std::stringstream ss;

        if (isVi) {
             ss << "<|system|>\nBạn là một chuyên gia trích xuất thông tin (OIE). Nhiệm vụ của bạn là đọc văn bản và trích xuất các mối quan hệ dưới dạng bộ ba (Chủ ngữ, Quan hệ, Tân ngữ).\n"
                << "Hãy trả về kết quả dưới dạng JSON:\n"
                << "[\n"
                << "  {\"source\": \"Chủ ngữ (Thực thể A)\", \"relation\": \"Mối quan hệ (động từ/tính từ)\", \"target\": \"Tân ngữ (Thực thể B)\", \"weight\": 1.0}\n"
                << "]\n\n"
                << "Văn bản:\n\"" << text << "\"\n"
                << "<|end|>\n<|assistant|>\n```json\n";
        } else {
            ss << "<|system|>\nYou are an Open Information Extraction (OIE) expert. Your task is to extract knowledge triples (Source, Relation, Target) from the text.\n"
               << "Return the result strictly as a JSON array of objects:\n"
               << "[\n"
               << "  {\"source\": \"Entity A\", \"relation\": \"relationship_to\", \"target\": \"Entity B\", \"weight\": 1.0}\n"
               << "]\n"
               << "Rules:\n"
               << "1. Use concise, active verbs for relations (e.g., \"loves\", \"attacks\", \"is_located_in\").\n"
               << "2. Ignore trivial or redundant information.\n"
               << "3. Weight should be 1.0 for facts, 0.5 for rumors.\n\n"
               << "Text:\n\"" << text << "\"\n"
               << "<|end|>\n<|assistant|>\n```json\n";
        }

        return ss.str();
    }

} // namespace NPCInference
