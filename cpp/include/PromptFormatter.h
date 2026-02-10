// PromptFormatter.h - Formats prompts to match Python training format
// Must exactly match the format from npc_cli.py lines 53-64

#pragma once

#include <string>

namespace NPCInference {

class PromptFormatter {
public:
    /**
     * Format prompt to match training data format:
     * "System: {persona}\nName: {npc_id}\nContext: {scenario}\n\nQuestion: {player_input}\nAnswer:"
     * 
     * @param persona NPC persona description
     * @param npc_id NPC identifier
     * @param scenario Current scenario/location
     * @param player_input Player's input text
     * @return Formatted prompt string
     */
    std::string Format(
        const std::string& persona,
        const std::string& npc_id,
        const std::string& scenario,
        const std::string& player_input
    ) const;
    
    /**
     * Parse JSON context and player_input into formatted prompt
     * Matches Python implementation from npc_cli.py
     * 
     * @param json_context JSON object with {npc_id, persona, scenario}
     * @param player_input Player's input text
     * @return Formatted prompt string
     */
    std::string FormatFromJSON(
        const std::string& json_context,
        const std::string& player_input
    ) const;
};

} // namespace NPCInference
