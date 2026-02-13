#include "PromptFormatter.h"
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

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
    try {
        auto j = json::parse(json_context);
        
        std::string persona = j.value("persona", "You are a helpful NPC.");
        std::string name = j.value("npc_id", "NPC");
        std::string scenario = j.value("scenario", "A mysterious land.");
        
        // If nested structure exists
        if (j.contains("context")) {
            auto& ctx = j["context"];
            persona = ctx.value("persona", persona);
            name = ctx.value("npc_id", name);
            scenario = ctx.value("scenario", scenario);
        }

        return Format(persona, name, scenario, player_input);
    } catch (...) {
        return "System: Error parsing context.\nQuestion: " + player_input + "\nAnswer:";
    }
}

} // namespace NPCInference
