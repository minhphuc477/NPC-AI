#pragma once

#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace NPCInference {

struct ResponseControlConfig {
    float min_context_coverage = 0.25f;
    float min_persona_coverage = 0.15f;
    float rewrite_temperature = 0.2f;
    int rewrite_max_tokens = 96;
    int rewrite_candidates = 3;
    float rewrite_temperature_step = 0.15f;
    bool enable_rewrite = true;
    bool allow_best_effort_rewrite = true;
};

struct ResponseControlResult {
    std::string response;
    std::string source;
    float context_coverage = 0.0f;
    float persona_coverage = 0.0f;
    bool repaired = false;
    std::string repair_reason;
};

class ResponseController {
public:
    using RewriteFn = std::function<std::string(const std::string&, int, float)>;

    static std::string SanitizeResponse(const std::string& text);

    static std::vector<std::string> ExtractContextKeywords(
        const nlohmann::json& context,
        size_t max_items = 12
    );

    static std::vector<std::string> ExtractPersonaKeywords(
        const std::string& persona,
        size_t max_items = 10
    );

    static std::string BuildDynamicContext(const nlohmann::json& context);

    static ResponseControlResult ControlResponse(
        const std::string& raw_response,
        const std::string& persona,
        const std::string& dynamic_context,
        const std::string& player_input,
        const std::vector<std::string>& context_keywords,
        const std::vector<std::string>& persona_keywords,
        const ResponseControlConfig& config,
        const RewriteFn& rewrite_fn = nullptr
    );
};

} // namespace NPCInference
