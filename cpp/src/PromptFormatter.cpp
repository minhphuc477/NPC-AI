// PromptFormatter.cpp - Implementation of prompt formatting

#include "PromptFormatter.h"
#include <sstream>

namespace NPCInference {

std::string PromptFormatter::Format(
    const std::string& persona,
    const std::string& npc_id,
    const std::string& scenario,
    const std::string& player_input
) const {
    // Match exact format from Python npc_cli.py:
    // prompt = f"System: {persona}\nName: {npc_id}\nContext: {scenario}\n\nQuestion: {player_input}\nAnswer:"
    
    std::ostringstream oss;
    oss << "System: " << persona << "\n"
        << "Name: " << npc_id << "\n"
        << "Context: " << scenario << "\n\n"
        << "Question: " << player_input << "\n"
        << "Answer:";
    
    return oss.str();
}

std::string PromptFormatter::FormatFromJSON(
    const std::string& json_context,
    const std::string& player_input
) const {
    // TODO: Parse JSON to extract persona, npc_id, scenario
    // For now, this is a placeholder that will be implemented
    // when we add JSON parsing (using nlohmann/json or similar)
    
    // Example JSON structure:
    // {
    //   "context": {
    //     "npc_id": "Guard_1",
    //     "persona": "You are a loyal guard...",
    //     "scenario": "Village gate"
    //   },
    //   "player_input": "Hello..."
    // }
    
    // Will be implemented in next iteration
    return "";
}

} // namespace NPCInference
