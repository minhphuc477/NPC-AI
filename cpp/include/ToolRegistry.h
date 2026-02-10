#pragma once

#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <memory>
#include <nlohmann/json.hpp>

namespace NPCInference {

using json = nlohmann::json;

/**
 * Tool/Function Calling Framework
 * 
 * Enables NPCs to call external functions and tools
 */
class ToolRegistry {
public:
    using ToolFunction = std::function<json(const json&)>;

    struct ToolSchema {
        std::string name;
        std::string description;
        json parameters_schema;  // JSON Schema for parameters
        ToolFunction function;
    };

    struct ToolCall {
        std::string tool_name;
        json arguments;
        json result;
        bool success = false;
        std::string error_message;
    };

    /**
     * Register a tool
     */
    void RegisterTool(const std::string& name,
                     const std::string& description,
                     const json& parameters_schema,
                     ToolFunction function);

    /**
     * Execute a tool call
     */
    ToolCall ExecuteTool(const std::string& tool_name, const json& arguments);

    /**
     * Get all registered tools as JSON schema
     */
    json GetToolsSchema() const;

    /**
     * Check if tool exists
     */
    bool HasTool(const std::string& name) const;

    /**
     * Get tool names
     */
    std::vector<std::string> GetToolNames() const;

private:
    std::unordered_map<std::string, ToolSchema> tools_;
};

/**
 * Built-in tools for NPCs
 */
class BuiltInTools {
public:
    static void RegisterAll(ToolRegistry& registry);

private:
    static json GetCurrentTime(const json& args);
    static json GetWeather(const json& args);
    static json SearchKnowledge(const json& args);
    static json RecallMemory(const json& args);
};

} // namespace NPCInference
