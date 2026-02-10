#pragma once

#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace NPCInference {

    class PromptBuilder {
    public:
        PromptBuilder(bool useAdvancedFormat = true);

        std::string Build(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& language = "vi", const json& tools = {});
        
    private:
        bool useAdvancedFormat_;

        std::string BuildAdvanced(const json& npcData, const json& gameState, const std::string& playerInput, const std::string& language, const json& tools);
        std::string BuildLegacy(const json& npcData, const json& gameState, const std::string& playerInput);
    };

} // namespace NPCInference
