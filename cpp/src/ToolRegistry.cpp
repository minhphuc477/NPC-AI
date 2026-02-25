#include "ToolRegistry.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <functional>

namespace NPCInference {

namespace {
BuiltInTools::Providers g_tool_providers;

bool IsTruthyEnv(const char* value) {
    if (!value) return false;
    const std::string s(value);
    return s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON";
}
} // namespace

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

    // Get weather
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

void BuiltInTools::SetProviders(const Providers& providers) {
    g_tool_providers = providers;
}

void BuiltInTools::ClearProviders() {
    g_tool_providers = Providers{};
}

bool BuiltInTools::HasExternalProviders() {
    return static_cast<bool>(g_tool_providers.weather_provider) ||
           static_cast<bool>(g_tool_providers.knowledge_provider) ||
           static_cast<bool>(g_tool_providers.memory_provider);
}

const BuiltInTools::Providers& BuiltInTools::GetProviders() {
    return g_tool_providers;
}

bool BuiltInTools::AllowSimulatedFallback() {
    return IsTruthyEnv(std::getenv("NPC_ALLOW_SIMULATED_TOOLS"));
}

json BuiltInTools::BuildUnavailableResult(const std::string& tool_name, const std::string& reason) {
    return json{
        {"tool", tool_name},
        {"available", false},
        {"reason", reason}
    };
}

json BuiltInTools::BuildSimulatedWeather(const std::string& location) {
    const std::size_t h = std::hash<std::string>{}(location);
    static const std::vector<std::string> kConditions = {
        "Clear",
        "Cloudy",
        "Light Rain",
        "Windy",
        "Overcast",
        "Partly Cloudy"
    };

    const int temp_c = static_cast<int>(8 + (h % 25));
    const int humidity = static_cast<int>(30 + (h % 61));
    const std::string condition = kConditions[h % kConditions.size()];

    return json{
        {"location", location},
        {"temperature_c", temp_c},
        {"condition", condition},
        {"humidity_percent", humidity},
        {"available", true},
        {"source", "simulated"},
        {"simulated", true}
    };
}

json BuiltInTools::BuildSimulatedKnowledge(const std::string& query) {
    return json{
        {"query", query},
        {"results", json::array({
            {
                {"title", "Simulated Knowledge Snippet"},
                {"snippet", "No external provider configured. Query retained for offline testing."}
            }
        })},
        {"available", true},
        {"source", "simulated"},
        {"simulated", true}
    };
}

json BuiltInTools::BuildSimulatedMemory(const std::string& topic) {
    return json{
        {"topic", topic},
        {"memories", json::array({
            "No persistent memory provider configured. This is a simulated memory response."
        })},
        {"available", true},
        {"source", "simulated"},
        {"simulated", true}
    };
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

    const auto& providers = GetProviders();
    if (providers.weather_provider) {
        json result = providers.weather_provider(location);
        if (!result.contains("source")) result["source"] = "provider";
        result["available"] = true;
        result["simulated"] = false;
        return result;
    }

    if (AllowSimulatedFallback()) {
        return BuildSimulatedWeather(location);
    }

    return BuildUnavailableResult("get_weather", "Weather provider not configured");
}

json BuiltInTools::SearchKnowledge(const json& args) {
    std::string query = args.value("query", "");

    const auto& providers = GetProviders();
    if (providers.knowledge_provider) {
        json result = providers.knowledge_provider(query);
        if (!result.contains("source")) result["source"] = "provider";
        result["available"] = true;
        result["simulated"] = false;
        return result;
    }

    if (AllowSimulatedFallback()) {
        return BuildSimulatedKnowledge(query);
    }

    return BuildUnavailableResult("search_knowledge", "Knowledge provider not configured");
}

json BuiltInTools::RecallMemory(const json& args) {
    std::string topic = args.value("topic", "");

    const auto& providers = GetProviders();
    if (providers.memory_provider) {
        json result = providers.memory_provider(topic);
        if (!result.contains("source")) result["source"] = "provider";
        result["available"] = true;
        result["simulated"] = false;
        return result;
    }

    if (AllowSimulatedFallback()) {
        return BuildSimulatedMemory(topic);
    }

    return BuildUnavailableResult("recall_memory", "Memory provider not configured");
}

} // namespace NPCInference
