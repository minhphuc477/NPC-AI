#include "../include/NPCInference.h"
#include "../include/ToolRegistry.h"
#include <iostream>

using namespace NPCInference;

int main() {
    std::cout << "=== Tool Execution Test ===" << std::endl;
    
    // Test 1: Create and register tools
    std::cout << "\n[Test 1] Creating tool registry" << std::endl;
    ToolRegistry registry;
    BuiltInTools::RegisterAll(registry);
    std::cout << "✓ Built-in tools registered" << std::endl;
    
    // Test 2: List registered tools
    std::cout << "\n[Test 2] Listing registered tools" << std::endl;
    auto tool_names = registry.GetToolNames();
    std::cout << "  Registered tools (" << tool_names.size() << "):" << std::endl;
    for (const auto& name : tool_names) {
        std::cout << "    - " << name << std::endl;
    }
    
    if (tool_names.size() >= 4) {
        std::cout << "✓ Expected number of tools registered" << std::endl;
    } else {
        std::cerr << "✗ Expected at least 4 tools" << std::endl;
        return 1;
    }
    
    // Test 3: Check tool existence
    std::cout << "\n[Test 3] Checking tool existence" << std::endl;
    if (registry.HasTool("get_current_time")) {
        std::cout << "✓ get_current_time tool found" << std::endl;
    } else {
        std::cerr << "✗ get_current_time tool not found" << std::endl;
        return 1;
    }
    
    // Test 4: Execute get_current_time
    std::cout << "\n[Test 4] Executing get_current_time" << std::endl;
    nlohmann::json empty_args = nlohmann::json::object();
    auto result = registry.ExecuteTool("get_current_time", empty_args);
    
    if (result.success) {
        std::cout << "✓ Tool executed successfully" << std::endl;
        std::cout << "  Result: " << result.result.dump(2) << std::endl;
        
        if (result.result.contains("datetime") && result.result.contains("timestamp")) {
            std::cout << "✓ Result contains expected fields" << std::endl;
        }
    } else {
        std::cerr << "✗ Tool execution failed: " << result.error_message << std::endl;
        return 1;
    }
    
    // Test 5: Execute get_weather with arguments
    std::cout << "\n[Test 5] Executing get_weather with arguments" << std::endl;
    nlohmann::json weather_args;
    weather_args["location"] = "Hanoi";
    
    auto weather_result = registry.ExecuteTool("get_weather", weather_args);
    if (weather_result.success) {
        std::cout << "✓ get_weather executed successfully" << std::endl;
        std::cout << "  Result: " << weather_result.result.dump(2) << std::endl;
        
        if (weather_result.result["location"] == "Hanoi") {
            std::cout << "✓ Location parameter correctly passed" << std::endl;
        }
    } else {
        std::cerr << "✗ get_weather failed: " << weather_result.error_message << std::endl;
        return 1;
    }
    
    // Test 6: Execute search_knowledge
    std::cout << "\n[Test 6] Executing search_knowledge" << std::endl;
    nlohmann::json search_args;
    search_args["query"] = "game lore";
    
    auto search_result = registry.ExecuteTool("search_knowledge", search_args);
    if (search_result.success) {
        std::cout << "✓ search_knowledge executed successfully" << std::endl;
        std::cout << "  Result: " << search_result.result.dump(2) << std::endl;
    }
    
    // Test 7: Execute recall_memory
    std::cout << "\n[Test 7] Executing recall_memory" << std::endl;
    nlohmann::json recall_args;
    recall_args["topic"] = "player interaction";
    
    auto recall_result = registry.ExecuteTool("recall_memory", recall_args);
    if (recall_result.success) {
        std::cout << "✓ recall_memory executed successfully" << std::endl;
        std::cout << "  Result: " << recall_result.result.dump(2) << std::endl;
    }
    
    // Test 8: Try to execute non-existent tool
    std::cout << "\n[Test 8] Testing error handling for non-existent tool" << std::endl;
    auto bad_result = registry.ExecuteTool("nonexistent_tool", empty_args);
    
    if (!bad_result.success) {
        std::cout << "✓ Correctly handled non-existent tool" << std::endl;
        std::cout << "  Error: " << bad_result.error_message << std::endl;
    } else {
        std::cerr << "✗ Should have failed on non-existent tool" << std::endl;
        return 1;
    }
    
    // Test 9: Get tools schema
    std::cout << "\n[Test 9] Getting tools schema" << std::endl;
    nlohmann::json schema = registry.GetToolsSchema();
    
    if (schema.is_array() && schema.size() >= 4) {
        std::cout << "✓ Tools schema generated" << std::endl;
        std::cout << "  Schema for first tool:" << std::endl;
        std::cout << schema[0].dump(2) << std::endl;
    } else {
        std::cerr << "✗ Invalid tools schema" << std::endl;
        return 1;
    }
    
    // Test 10: Register custom tool
    std::cout << "\n[Test 10] Registering custom tool" << std::endl;
    registry.RegisterTool(
        "custom_test",
        "A custom test tool",
        nlohmann::json{
            {"type", "object"},
            {"properties", nlohmann::json::object()},
            {"required", nlohmann::json::array()}
        },
        [](const nlohmann::json& args) -> nlohmann::json {
            return nlohmann::json{{"message", "Custom tool executed"}};
        }
    );
    
    if (registry.HasTool("custom_test")) {
        std::cout << "✓ Custom tool registered" << std::endl;
        
        auto custom_result = registry.ExecuteTool("custom_test", empty_args);
        if (custom_result.success && custom_result.result["message"] == "Custom tool executed") {
            std::cout << "✓ Custom tool executed successfully" << std::endl;
        }
    }
    
    std::cout << "\n=== All Tool Execution Tests Passed ===" << std::endl;
    return 0;
}
