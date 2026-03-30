#include "ToolRegistry.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void SetEnvVar(const std::string& key, const std::string& value) {
#ifdef _WIN32
    _putenv_s(key.c_str(), value.c_str());
#else
    setenv(key.c_str(), value.c_str(), 1);
#endif
}

void UnsetEnvVar(const std::string& key) {
#ifdef _WIN32
    _putenv_s(key.c_str(), "");
#else
    unsetenv(key.c_str());
#endif
}

} // namespace

int main() {
    using namespace NPCInference;

    BuiltInTools::ClearProviders();
    UnsetEnvVar("NPC_ALLOW_SIMULATED_TOOLS");

    ToolRegistry registry;
    BuiltInTools::RegisterAll(registry);

    auto unavailable_weather = registry.ExecuteTool("get_weather", {{"location", "Hanoi"}});
    assert(unavailable_weather.success);
    assert(unavailable_weather.result.value("available", true) == false);

    SetEnvVar("NPC_ALLOW_SIMULATED_TOOLS", "1");
    auto simulated_weather = registry.ExecuteTool("get_weather", {{"location", "Hanoi"}});
    assert(simulated_weather.success);
    assert(simulated_weather.result.value("simulated", false) == true);
    assert(simulated_weather.result.value("available", false) == true);

    BuiltInTools::Providers providers;
    providers.weather_provider = [](const std::string& location) {
        return nlohmann::json{
            {"location", location},
            {"condition", "ProviderClear"},
            {"temperature_c", 26}
        };
    };
    BuiltInTools::SetProviders(providers);
    UnsetEnvVar("NPC_ALLOW_SIMULATED_TOOLS");

    auto provider_weather = registry.ExecuteTool("get_weather", {{"location", "Hue"}});
    assert(provider_weather.success);
    assert(provider_weather.result.value("source", "") == "provider");
    assert(provider_weather.result.value("simulated", true) == false);
    assert(provider_weather.result.value("condition", "") == "ProviderClear");

    BuiltInTools::ClearProviders();
    UnsetEnvVar("NPC_ALLOW_SIMULATED_TOOLS");

    std::cout << "ToolRegistry provider/simulation behavior test PASSED." << std::endl;
    return 0;
}
