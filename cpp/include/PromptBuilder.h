#pragma once

#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace NPCInference {

    class PromptBuilder {
    public:
        PromptBuilder(bool useAdvancedFormat = true, bool useJsonFormat = false);

        std::string Build(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& language = "vi", const json& tools = {});
        
        // Phase 2: Planner-Executor
        std::string BuildPlanning(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& language = "vi");
        std::string BuildWithThought(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& thought, const std::string& language = "vi", const json& tools = {});
        
        // Phase 2: Reflection Engine
        std::string BuildCritique(const std::string& originalResponse, const json& npcData, const json& gameState, const std::string& language = "vi");
        std::string BuildRefine(const std::string& originalResponse, const std::string& critique, const json& npcData, const json& gameState, const std::string& language = "vi");

        // Phase 2: Neuro-symbolic Truth Guard
        std::string BuildTruthGuardCheck(const std::string& response, const std::string& worldFacts, const std::string& language = "vi");

    private:
        bool useAdvancedFormat_;
        bool useJsonFormat_;

        std::string BuildAdvanced(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& language, const json& tools);
        std::string BuildLegacy(const json& npcData, const json& gameState, const std::string& playerInput);
    };

} // namespace NPCInference
