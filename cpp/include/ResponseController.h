#pragma once

#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace NPCInference {

struct ResponseControlConfig {
    float min_context_coverage = 0.33f;
    float min_persona_coverage = 0.18f;
    int min_response_tokens = 8;
    float rewrite_temperature = 0.2f;
    int rewrite_max_tokens = 96;
    int rewrite_candidates = 2;
    float rewrite_temperature_step = 0.15f;
    bool early_stop_on_pass = true;
    float early_stop_score = 0.70f;
    bool allow_relaxed_acceptance = true;
    float relaxed_context_coverage = 0.18f;
    float relaxed_persona_coverage = 0.09f;
    float relaxed_candidate_score = 0.44f;
    float min_rewrite_gain = 0.015f;
    bool enable_rewrite = true;
    bool allow_best_effort_rewrite = true;
    bool behavior_adaptation_enabled = true;
    bool adaptive_acceptance_enabled = true;
    float adaptive_candidate_score = 0.38f;
    float adaptive_context_coverage = 0.14f;
    float adaptive_persona_coverage = 0.10f;
    float adaptive_high_confidence_score = 0.53f;
    float adaptive_mid_confidence_score = 0.40f;
    int adaptive_high_confidence_rewrites = 1;
    int adaptive_mid_confidence_rewrites = 2;
    int adaptive_low_confidence_rewrites = 3;
    bool low_confidence_retry_requires_gain = true;
    float low_confidence_retry_min_score_gain = 0.01f;
    float low_confidence_retry_min_coverage_gain = 0.02f;
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
