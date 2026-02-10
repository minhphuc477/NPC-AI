#include "ToolRegistry.h"
#include <iostream>
#include <chrono>
#include <ctime>

namespace NPCInference {

void ToolRegistry::RegisterTool(const std::string& name,
                                const std::string& description,
                                const json& parameters_schema,
                                ToolFunction function) {
    ToolSchema schema;
    schema.name = name;
    schema.description = description;
    schema.parameters_schema = parameters_schema;
    schema.function = function;
    
    tools_[name] = schema;
    std::cerr << "ToolRegistry: Registered tool '" << name << "'" << std::endl;
}

ToolRegistry::ToolCall ToolRegistry::ExecuteTool(const std::string& tool_name, 
                                                  const json& arguments) {
    ToolCall call;
    call.tool_name = tool_name;
    call.arguments = arguments;

    auto it = tools_.find(tool_name);
    if (it == tools_.end()) {
        call.success = false;
        call.error_message = "Tool not found: " + tool_name;
        return call;
    }

    try {
        call.result = it->second.function(arguments);
        call.success = true;
    } catch (const std::exception& e) {
        call.success = false;
        call.error_message = std::string("Tool execution failed: ") + e.what();
    }

    return call;
}

json ToolRegistry::GetToolsSchema() const {
    json tools_array = json::array();

    for (const auto& [name, schema] : tools_) {
        json tool_json = {
            {"name", schema.name},
            {"description", schema.description},
            {"parameters", schema.parameters_schema}
        };
        tools_array.push_back(tool_json);
    }

    return tools_array;
}

bool ToolRegistry::HasTool(const std::string& name) const {
    return tools_.find(name) != tools_.end();
}

std::vector<std::string> ToolRegistry::GetToolNames() const {
    std::vector<std::string> names;
    names.reserve(tools_.size());
    for (const auto& [name, _] : tools_) {
        names.push_back(name);
    }
    return names;
}

// Built-in tools implementation

void BuiltInTools::RegisterAll(ToolRegistry& registry) {
    // Get current time
    registry.RegisterTool(
        "get_current_time",
        "Get the current date and time",
        json{
            {"type", "object"},
            {"properties", json::object()},
            {"required", json::array()}
        },
        GetCurrentTime
    );

    // Get weather (mock)
    registry.RegisterTool(
        "get_weather",
        "Get weather information for a location",
        json{
            {"type", "object"},
            {"properties", {
                {"location", {{"type", "string"}, {"description", "City name"}}}
            }},
            {"required", json::array({"location"})}
        },
        GetWeather
    );

    // Search knowledge
    registry.RegisterTool(
        "search_knowledge",
        "Search the knowledge base",
        json{
            {"type", "object"},
            {"properties", {
                {"query", {{"type", "string"}, {"description", "Search query"}}}
            }},
            {"required", json::array({"query"})}
        },
        SearchKnowledge
    );

    // Recall memory
    registry.RegisterTool(
        "recall_memory",
        "Recall past memories",
        json{
            {"type", "object"},
            {"properties", {
                {"topic", {{"type", "string"}, {"description", "Memory topic"}}}
            }},
            {"required", json::array({"topic"})}
        },
        RecallMemory
    );
}

json BuiltInTools::GetCurrentTime(const json& args) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));
    
    return json{
        {"datetime", buffer},
        {"timestamp", now_c}
    };
}

json BuiltInTools::GetWeather(const json& args) {
    std::string location = args.value("location", "Unknown");
    
    // Mock weather data
    return json{
        {"location", location},
        {"temperature", 22},
        {"condition", "Partly Cloudy"},
        {"humidity", 65}
    };
}

json BuiltInTools::SearchKnowledge(const json& args) {
    std::string query = args.value("query", "");
    
    // Mock search results
    return json{
        {"query", query},
        {"results", json::array({
            {{"title", "Result 1"}, {"snippet", "Information about " + query}},
            {{"title", "Result 2"}, {"snippet", "More details on " + query}}
        })}
    };
}

json BuiltInTools::RecallMemory(const json& args) {
    std::string topic = args.value("topic", "");
    
    // Mock memory recall
    return json{
        {"topic", topic},
        {"memories", json::array({
            "I remember discussing " + topic + " before",
            "There was an important event related to " + topic
        })}
    };
}

} // namespace NPCInference
